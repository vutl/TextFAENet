from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import QaTaCOV19Dataset
from scripts.qata_vis_utils import gray_to_rgb, load_run, predict_one, sample_dice


PALETTE = {
    "ours": (57, 134, 108),
    "faenet": (206, 111, 78),
    "fmiseg": (67, 112, 184),
    "medclip": (146, 104, 178),
    "gt": (48, 154, 112),
    "tp": (52, 150, 112),
    "fp": (66, 133, 190),
    "fn": (220, 74, 67),
    "text": (29, 34, 42),
    "muted": (93, 103, 117),
    "line": (222, 226, 232),
}


@dataclass
class ModelSpec:
    label: str
    kind: str
    color: tuple[int, int, int]
    run_dir: Path | None = None
    metric_csv: Path | None = None
    pred_dir: Path | None = None


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[Path] = []
    if sys.platform.startswith("win"):
        candidates.extend(
            [
                Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            ]
        )
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(26, True)
FONT_HEADER = font(16, True)
FONT_LABEL = font(13, True)
FONT_SMALL = font(12)


def parse_color(value: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not value:
        return fallback
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid color: {value}")
    return tuple(parts)  # type: ignore[return-value]


def parse_text_spec(raw: str) -> ModelSpec:
    # Label=run_dir[=#rrggbb]
    parts = raw.split("=")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Text spec must be Label=run_dir[=#rrggbb], got: {raw}")
    label, run_dir = parts[:2]
    fallback = PALETTE["ours"] if label.lower().startswith("ours") else PALETTE["faenet"]
    return ModelSpec(label=label, kind="textfaenet", run_dir=(ROOT / run_dir).resolve(), color=parse_color(parts[2] if len(parts) == 3 else None, fallback))


def parse_external_spec(raw: str) -> ModelSpec:
    # Label=metric_csv=pred_dir[=#rrggbb]
    parts = raw.split("=")
    if len(parts) not in {3, 4}:
        raise ValueError(f"External spec must be Label=metric_csv=pred_dir[=#rrggbb], got: {raw}")
    label, metric_csv, pred_dir = parts[:3]
    lower = label.lower()
    fallback = PALETTE["fmiseg"] if "fmi" in lower else PALETTE["medclip"]
    return ModelSpec(
        label=label,
        kind="external_mask",
        metric_csv=(ROOT / metric_csv).resolve(),
        pred_dir=(ROOT / pred_dir).resolve(),
        color=parse_color(parts[3] if len(parts) == 4 else None, fallback),
    )


def load_metric_csv(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("mask_name") or row.get("name") or row.get("filename")
            if not name:
                continue
            rows[Path(name).name] = {
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "target_pixels": float(row.get("target_pixels", row.get("gt_pixels", 0.0)) or 0.0),
                "pred_pixels": float(row.get("pred_pixels", 0.0) or 0.0),
            }
    return rows


def dataset_index(ds: QaTaCOV19Dataset) -> dict[str, int]:
    out = {}
    for i in range(len(ds)):
        sample = ds[i]
        out[str(sample["mask_name"])] = i
    return out


def load_external_pred(pred_dir: Path, mask_name: str, shape: tuple[int, int]) -> np.ndarray:
    path = pred_dir / mask_name
    if not path.exists():
        stem = Path(mask_name).stem
        matches = list(pred_dir.glob(stem + ".*"))
        if not matches:
            raise FileNotFoundError(f"Missing external prediction mask: {path}")
        path = matches[0]
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = base.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray, base: np.ndarray) -> np.ndarray:
    canvas = (base.astype(np.float32) * 0.58 + 255.0 * 0.42).clip(0, 255).astype(np.uint8)
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    canvas[pred & gt] = np.array(PALETTE["tp"], dtype=np.uint8)
    canvas[pred & ~gt] = np.array(PALETTE["fp"], dtype=np.uint8)
    canvas[(~pred) & gt] = np.array(PALETTE["fn"], dtype=np.uint8)
    return canvas


def resize_rgb(arr: np.ndarray, tile: int) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB").resize((tile, tile), Image.BILINEAR)


def draw_tile(
    out: Image.Image,
    draw: ImageDraw.ImageDraw,
    arr: np.ndarray,
    x: int,
    y: int,
    tile: int,
    score: str | None = None,
) -> None:
    draw.rounded_rectangle((x - 3, y - 3, x + tile + 3, y + tile + 3), radius=8, fill=(248, 249, 247), outline=PALETTE["line"], width=1)
    out.paste(resize_rgb(arr, tile), (x, y))
    if score:
        w = 58
        draw.rounded_rectangle((x + tile - w - 7, y + 7, x + tile - 7, y + 29), radius=7, fill=(255, 255, 255), outline=(205, 210, 217), width=1)
        draw.text((x + tile - w + 2, y + 10), score, fill=PALETTE["text"], font=FONT_SMALL)


def select_cases(
    metrics: dict[str, dict[str, dict[str, float]]],
    ds_index: dict[str, int],
    target_label: str,
    labels: list[str],
    num_cases: int,
    min_target_dice: float,
    min_margin: float,
    min_baseline_dice: float,
    max_other_dice: float | None,
    min_target_pixels: float,
    max_target_pixels: float,
) -> list[dict]:
    common = set(ds_index)
    for label in labels:
        common &= set(metrics[label])
    rows = []
    for mask_name in common:
        target = metrics[target_label][mask_name]
        target_pixels = float(target.get("target_pixels", 0.0))
        if target_pixels < min_target_pixels or target_pixels > max_target_pixels:
            continue
        baseline_scores = [float(metrics[label][mask_name]["dice"]) for label in labels if label != target_label]
        other = max(baseline_scores)
        if baseline_scores and min(baseline_scores) < min_baseline_dice:
            continue
        target_dice = float(target["dice"])
        margin = target_dice - other
        if max_other_dice is not None and other > max_other_dice:
            continue
        rows.append(
            {
                "mask_name": mask_name,
                "dataset_index": ds_index[mask_name],
                "target_dice": target_dice,
                "second_dice": other,
                "margin": margin,
                "target_pixels": target_pixels,
            }
        )

    winners = [r for r in rows if r["target_dice"] >= min_target_dice and r["margin"] >= min_margin]
    if len(winners) < num_cases:
        winners = [r for r in rows if r["target_dice"] >= min_target_dice and r["margin"] > 0.0]
    if len(winners) < num_cases:
        winners = [r for r in rows if r["margin"] > 0.0]
    if len(winners) < num_cases:
        winners = rows

    return sorted(winners, key=lambda r: (r["margin"], r["target_dice"], r["target_pixels"]), reverse=True)[:num_cases]


@torch.inference_mode()
def collect_predictions(
    specs: list[ModelSpec],
    selected: list[dict],
    ds: QaTaCOV19Dataset,
    device: torch.device,
    threshold_override: float | None = None,
) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for spec in specs:
        print(f"Collecting predictions: {spec.label}", flush=True)
        model_preds: dict[str, dict] = {}
        if spec.kind == "textfaenet":
            assert spec.run_dir is not None
            model, tokenizer, cfg, threshold = load_run(spec.run_dir, device)
            if threshold_override is not None:
                threshold = threshold_override
            for row in selected:
                sample = ds[row["dataset_index"]]
                gt = sample["mask"].squeeze(0).numpy() > 0.5
                result = predict_one(model, tokenizer, cfg, sample["image"], str(sample["text"]), device, threshold=threshold)
                model_preds[row["mask_name"]] = {
                    "pred": result["pred"].astype(np.float32),
                    "dice": float(sample_dice(result["pred"], gt)),
                    "threshold": float(threshold),
                }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            assert spec.pred_dir is not None
            for row in selected:
                sample = ds[row["dataset_index"]]
                gt = sample["mask"].squeeze(0).numpy() > 0.5
                pred = load_external_pred(spec.pred_dir, row["mask_name"], gt.shape)
                model_preds[row["mask_name"]] = {
                    "pred": pred,
                    "dice": float(sample_dice(pred, gt)),
                    "threshold": 0.5,
                }
        out[spec.label] = model_preds
    return out


def render(
    ds: QaTaCOV19Dataset,
    specs: list[ModelSpec],
    selected: list[dict],
    preds: dict[str, dict[str, dict]],
    target_label: str,
    out_png: Path,
    out_json: Path,
    tile: int,
    title: str,
    subtitle: str,
    row_labels: bool,
    show_gt_area: bool,
) -> None:
    headers = ["Input", "GT", *[spec.label for spec in specs], f"Error ({target_label})"]
    gap = 12
    margin = 28
    label_w = 42 if row_labels else 0
    top = 76 if title or subtitle else 36
    row_gap = 28
    row_h = tile + row_gap
    width = margin * 2 + label_w + len(headers) * tile + (len(headers) - 1) * gap
    height = top + row_h * len(selected) + 48
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if title:
        draw.text((margin, 18), title, fill=PALETTE["text"], font=FONT_TITLE)
    if subtitle:
        draw.text((margin, 52), subtitle, fill=PALETTE["muted"], font=FONT_SMALL)

    x0 = margin + label_w
    for ci, header in enumerate(headers):
        draw.text((x0 + ci * (tile + gap) + 3, top - 24), header, fill=PALETTE["text"], font=FONT_HEADER)

    records = []
    target_spec = next(spec for spec in specs if spec.label == target_label)
    for ri, row in enumerate(selected, start=1):
        y = top + (ri - 1) * row_h
        sample = ds[row["dataset_index"]]
        base = gray_to_rgb(sample["image"].squeeze(0).numpy())
        gt = sample["mask"].squeeze(0).numpy() > 0.5
        if row_labels:
            draw.text((margin, y + tile // 2 - 9), f"C{ri}", fill=PALETTE["text"], font=FONT_LABEL)

        x = x0
        draw_tile(img, draw, base, x, y, tile)
        x += tile + gap
        gt_score = f"{gt.mean()*100:.1f}%" if show_gt_area else None
        draw_tile(img, draw, overlay_mask(base, gt, PALETTE["gt"], alpha=0.55), x, y, tile, score=gt_score)
        x += tile + gap

        scores = {}
        for spec in specs:
            pred = preds[spec.label][row["mask_name"]]["pred"] > 0.5
            dice = float(preds[spec.label][row["mask_name"]]["dice"])
            scores[spec.label] = dice
            draw_tile(img, draw, overlay_mask(base, pred, spec.color, alpha=0.50), x, y, tile, score=f"D {dice:.2f}")
            x += tile + gap

        target_pred = preds[target_label][row["mask_name"]]["pred"] > 0.5
        draw_tile(img, draw, error_map(target_pred, gt, base), x, y, tile)
        records.append(
            {
                "case_id": ri,
                "mask_name": row["mask_name"],
                "dataset_index": row["dataset_index"],
                "prompt": str(sample["text"]),
                "target_label": target_label,
                "selection_margin": float(row["margin"]),
                "target_pixels": float(row["target_pixels"]),
                "gt_area_ratio": float(gt.mean()),
                "dice": scores,
            }
        )

    ly = height - 30
    draw.text((margin, ly), "Error:", fill=PALETTE["muted"], font=FONT_SMALL)
    lx = margin + 52
    for label, color in [("TP", PALETTE["tp"]), ("FP", PALETTE["fp"]), ("FN", PALETTE["fn"])]:
        draw.rounded_rectangle((lx, ly, lx + 16, ly + 16), radius=3, fill=color)
        draw.text((lx + 22, ly - 1), label, fill=PALETTE["muted"], font=FONT_SMALL)
        lx += 66

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    img.save(out_png.with_suffix(".pdf"), "PDF", resolution=300.0)
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Make a fair QaTa qualitative figure against external baselines.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2"))
    parser.add_argument(
        "--text-run",
        action="append",
        default=None,
        help="Text-FAENet run spec: Label=run_dir[=#rrggbb]. Can be repeated.",
    )
    parser.add_argument(
        "--external",
        action="append",
        default=None,
        help="External mask baseline spec: Label=metric_csv=pred_dir[=#rrggbb]. Can be repeated.",
    )
    parser.add_argument("--target-label", default="Ours")
    parser.add_argument("--model-order", default="FAENet,FMISeg,MedCLIP-SAMv2,Ours")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--tile", type=int, default=152)
    parser.add_argument("--title", default="QaTa-COV19 qualitative comparison")
    parser.add_argument("--subtitle", default="Cases are selected where the target model has the strongest per-image Dice margin.")
    parser.add_argument("--no-row-labels", action="store_true")
    parser.add_argument("--show-gt-area", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-target-dice", type=float, default=0.78)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument(
        "--min-baseline-dice",
        type=float,
        default=0.0,
        help="Optional fairness filter: every loaded baseline must have at least this Dice on selected cases.",
    )
    parser.add_argument(
        "--max-other-dice",
        type=float,
        default=None,
        help="Optional contrast filter: reject cases where any non-target model exceeds this Dice.",
    )
    parser.add_argument("--min-target-pixels", type=float, default=250.0)
    parser.add_argument("--max-target-pixels", type=float, default=25000.0)
    parser.add_argument("--threshold", type=float, default=None, help="Optional override threshold for Text-FAENet runs.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "generated_figures" / "qata_external_qualitative")
    args = parser.parse_args()
    if args.subtitle is not None and args.subtitle.strip().lower() in {"", "none", "null", "-"}:
        args.subtitle = ""
    if args.title is not None and args.title.strip().lower() in {"", "none", "null", "-"}:
        args.title = ""
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir

    text_runs = args.text_run or [
        "FAENet=runs/qata_paper0516_qata_faenet_visual_clean_seed42=#ce6f4e",
        "Ours=runs/qata_paper0516_qata_simple_native_zero_both_seed42=#39866c",
    ]
    external_runs = args.external or [
        "FMISeg=external_metrics/fmiseg_qata_official/test_per_image_metrics.csv=external_metrics/fmiseg_qata_official/pred_masks_text_orientation=#4370b8",
        "MedCLIP-SAMv2=external_metrics/medclipsamv2_qata_our_prompt/test_per_image_metrics.csv=external_metrics/medclipsamv2_qata_our_prompt/pred_masks=#9268b2",
    ]
    all_specs = [parse_text_spec(x) for x in text_runs]
    all_specs.extend(parse_external_spec(x) for x in external_runs)

    loaded_specs: list[ModelSpec] = []
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    missing: list[str] = []
    for spec in all_specs:
        if spec.kind == "textfaenet":
            assert spec.run_dir is not None
            metric_csv = spec.run_dir / "test_per_image_metrics.csv"
            if not spec.run_dir.exists() or not metric_csv.exists():
                missing.append(f"{spec.label}: missing run or metrics at {metric_csv}")
                continue
            spec.metric_csv = metric_csv
        else:
            assert spec.metric_csv is not None and spec.pred_dir is not None
            if not spec.metric_csv.exists() or not spec.pred_dir.exists():
                missing.append(f"{spec.label}: missing metric CSV or pred dir")
                continue
        assert spec.metric_csv is not None
        metrics[spec.label] = load_metric_csv(spec.metric_csv)
        loaded_specs.append(spec)

    order = [x.strip() for x in args.model_order.split(",") if x.strip()]
    spec_by_label = {spec.label: spec for spec in loaded_specs}
    specs = [spec_by_label[label] for label in order if label in spec_by_label]
    specs.extend(spec for spec in loaded_specs if spec.label not in {s.label for s in specs})
    if args.target_label not in {spec.label for spec in specs}:
        raise RuntimeError(f"Target label `{args.target_label}` is not loaded. Missing: {missing}")
    if len(specs) < 2:
        raise RuntimeError(f"Need at least target + one baseline. Missing: {missing}")

    ds = QaTaCOV19Dataset(root_dir=str(args.data_root), split="test", image_size=224, use_text=True)
    idx = dataset_index(ds)
    selected = select_cases(
        metrics=metrics,
        ds_index=idx,
        target_label=args.target_label,
        labels=[spec.label for spec in specs],
        num_cases=args.num_cases,
        min_target_dice=args.min_target_dice,
        min_margin=args.min_margin,
        min_baseline_dice=args.min_baseline_dice,
        max_other_dice=args.max_other_dice,
        min_target_pixels=args.min_target_pixels,
        max_target_pixels=args.max_target_pixels,
    )
    if not selected:
        raise RuntimeError(f"No selectable cases. Missing: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = collect_predictions(specs, selected, ds, device, threshold_override=args.threshold)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render(
        ds=ds,
        specs=specs,
        selected=selected,
        preds=predictions,
        target_label=args.target_label,
        out_png=args.out_dir / "fig_qata_external_qualitative.png",
        out_json=args.out_dir / "fig_qata_external_qualitative_cases.json",
        tile=args.tile,
        title=args.title,
        subtitle=args.subtitle,
        row_labels=not args.no_row_labels,
        show_gt_area=args.show_gt_area,
    )
    manifest = {
        "loaded_models": [spec.label for spec in specs],
        "missing_models": missing,
        "target_label": args.target_label,
        "figure": str((args.out_dir / "fig_qata_external_qualitative.png").relative_to(ROOT)),
        "cases": str((args.out_dir / "fig_qata_external_qualitative_cases.json").relative_to(ROOT)),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        print("Skipped missing baselines:", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
