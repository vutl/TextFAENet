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

from src.data.mosmed_text_csv import MosMedTextCSVDataset
from scripts.qata_vis_utils import gray_to_rgb, load_run, predict_one, sample_dice


PALETTE = {
    "ours": (57, 134, 108),
    "text": (67, 112, 184),
    "faenet": (206, 111, 78),
    "gt": (48, 154, 112),
    "tp": (52, 150, 112),
    "fp": (66, 133, 190),
    "fn": (220, 74, 67),
    "ink": (29, 34, 42),
    "muted": (93, 103, 117),
    "line": (222, 226, 232),
}


@dataclass
class RunSpec:
    label: str
    run_dir: Path
    color: tuple[int, int, int]


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


def parse_run_spec(raw: str) -> RunSpec:
    # Label=run_dir[=#rrggbb]
    parts = raw.split("=")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Run spec must be Label=run_dir[=#rrggbb], got: {raw}")
    label, run_dir = parts[:2]
    lower = label.lower()
    fallback = PALETTE["ours"] if "ours" in lower else (PALETTE["faenet"] if "faenet" in lower else PALETTE["text"])
    return RunSpec(label=label, run_dir=(ROOT / run_dir).resolve(), color=parse_color(parts[2] if len(parts) == 3 else None, fallback))


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.50) -> np.ndarray:
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
        draw.text((x + tile - w + 2, y + 10), score, fill=PALETTE["ink"], font=FONT_SMALL)


@torch.inference_mode()
def evaluate_runs(
    ds: MosMedTextCSVDataset,
    specs: list[RunSpec],
    device: torch.device,
    threshold_override: float | None,
) -> dict[str, dict[str, dict]]:
    all_preds: dict[str, dict[str, dict]] = {}
    for spec in specs:
        print(f"Evaluating {spec.label}: {spec.run_dir}", flush=True)
        model, tokenizer, cfg, threshold = load_run(spec.run_dir, device)
        if threshold_override is not None:
            threshold = threshold_override
        rows = {}
        for idx in range(len(ds)):
            sample = ds[idx]
            gt = sample["mask"].squeeze(0).numpy() > 0.5
            result = predict_one(model, tokenizer, cfg, sample["image"], str(sample["text"]), device, threshold=threshold)
            pred = result["pred"] > 0.5
            rows[str(sample["mask_name"])] = {
                "dataset_index": idx,
                "pred": pred.astype(np.float32),
                "dice": float(sample_dice(pred, gt)),
                "iou": float(((pred & gt).sum() + 1e-6) / (((pred | gt).sum()) + 1e-6)),
                "target_pixels": float(gt.sum()),
                "pred_pixels": float(pred.sum()),
                "threshold": float(threshold),
            }
        all_preds[spec.label] = rows
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_preds


def select_cases(
    preds: dict[str, dict[str, dict]],
    target_label: str,
    labels: list[str],
    num_cases: int,
    min_target_dice: float,
    min_margin: float,
    min_target_pixels: float,
    max_target_pixels: float,
) -> list[dict]:
    common = set(preds[target_label])
    for label in labels:
        common &= set(preds[label])
    rows = []
    for name in common:
        target = preds[target_label][name]
        target_pixels = float(target["target_pixels"])
        if target_pixels < min_target_pixels or target_pixels > max_target_pixels:
            continue
        other = max(float(preds[label][name]["dice"]) for label in labels if label != target_label)
        rows.append(
            {
                "mask_name": name,
                "dataset_index": int(target["dataset_index"]),
                "target_dice": float(target["dice"]),
                "second_dice": other,
                "margin": float(target["dice"] - other),
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


def write_metrics_csv(preds: dict[str, dict[str, dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, rows in preds.items():
        safe = label.lower().replace(" ", "_").replace("-", "_")
        with (out_dir / f"{safe}_per_image_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mask_name", "dice", "iou", "target_pixels", "pred_pixels", "threshold"])
            writer.writeheader()
            for name, row in sorted(rows.items()):
                writer.writerow(
                    {
                        "mask_name": name,
                        "dice": row["dice"],
                        "iou": row["iou"],
                        "target_pixels": row["target_pixels"],
                        "pred_pixels": row["pred_pixels"],
                        "threshold": row["threshold"],
                    }
                )


def render(
    ds: MosMedTextCSVDataset,
    specs: list[RunSpec],
    selected: list[dict],
    preds: dict[str, dict[str, dict]],
    target_label: str,
    out_png: Path,
    out_json: Path,
    tile: int,
    title: str,
    subtitle: str,
    row_labels: bool,
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
        draw.text((margin, 18), title, fill=PALETTE["ink"], font=FONT_TITLE)
    if subtitle:
        draw.text((margin, 52), subtitle, fill=PALETTE["muted"], font=FONT_SMALL)

    x0 = margin + label_w
    for ci, header in enumerate(headers):
        draw.text((x0 + ci * (tile + gap) + 3, top - 24), header, fill=PALETTE["ink"], font=FONT_HEADER)

    records = []
    for ri, row in enumerate(selected, start=1):
        y = top + (ri - 1) * row_h
        sample = ds[row["dataset_index"]]
        base = gray_to_rgb(sample["image"].squeeze(0).numpy())
        gt = sample["mask"].squeeze(0).numpy() > 0.5
        if row_labels:
            draw.text((margin, y + tile // 2 - 9), f"C{ri}", fill=PALETTE["ink"], font=FONT_LABEL)

        x = x0
        draw_tile(img, draw, base, x, y, tile)
        x += tile + gap
        draw_tile(img, draw, overlay_mask(base, gt, PALETTE["gt"], alpha=0.55), x, y, tile, score=f"{gt.mean()*100:.1f}%")
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
    parser = argparse.ArgumentParser("Make MosMed qualitative comparison figure.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets" / "MosMed")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run spec: Label=run_dir[=#rrggbb]. Can be repeated.",
    )
    parser.add_argument("--target-label", default="Ours")
    parser.add_argument("--model-order", default="FAENet,Text-FAENet,Ours")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--tile", type=int, default=152)
    parser.add_argument("--min-target-dice", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--min-target-pixels", type=float, default=120.0)
    parser.add_argument("--max-target-pixels", type=float, default=18000.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--title", default="MosMed qualitative comparison")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--no-row-labels", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "generated_figures" / "mosmed_qualitative")
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir

    run_specs = args.run or [
        "FAENet=runs/batch20260422_mosmed_faenet_notext_e50=#ce6f4e",
        "Text-FAENet=runs/mosmed_text_faenet_v2_suite_smoke_fix=#4370b8",
        "Ours=runs/screening0506_mosmed_cxr_frozen_keep_both_seed42=#39866c",
    ]
    specs = [parse_run_spec(x) for x in run_specs]
    spec_by_label = {spec.label: spec for spec in specs}
    order = [x.strip() for x in args.model_order.split(",") if x.strip()]
    specs = [spec_by_label[label] for label in order if label in spec_by_label]
    specs.extend(spec for spec in spec_by_label.values() if spec.label not in {s.label for s in specs})
    if args.target_label not in {spec.label for spec in specs}:
        raise RuntimeError(f"Target label `{args.target_label}` is not in loaded specs.")

    ds = MosMedTextCSVDataset(root_dir=str(args.data_root), split="test", image_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = evaluate_runs(ds, specs, device, threshold_override=args.threshold)
    selected = select_cases(
        preds=preds,
        target_label=args.target_label,
        labels=[spec.label for spec in specs],
        num_cases=args.num_cases,
        min_target_dice=args.min_target_dice,
        min_margin=args.min_margin,
        min_target_pixels=args.min_target_pixels,
        max_target_pixels=args.max_target_pixels,
    )
    if not selected:
        raise RuntimeError("No selectable MosMed cases.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(preds, args.out_dir)
    render(
        ds=ds,
        specs=specs,
        selected=selected,
        preds=preds,
        target_label=args.target_label,
        out_png=args.out_dir / "fig_mosmed_qualitative.png",
        out_json=args.out_dir / "fig_mosmed_qualitative_cases.json",
        tile=args.tile,
        title=args.title,
        subtitle=args.subtitle,
        row_labels=not args.no_row_labels,
    )
    manifest = {
        "loaded_models": [spec.label for spec in specs],
        "target_label": args.target_label,
        "figure": str((args.out_dir / "fig_mosmed_qualitative.png").relative_to(ROOT)),
        "cases": str((args.out_dir / "fig_mosmed_qualitative_cases.json").relative_to(ROOT)),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
