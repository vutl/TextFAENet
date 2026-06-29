from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import QaTaCOV19Dataset
from scripts.qata_vis_utils import gray_to_rgb, load_run, predict_one, sample_dice


RUN_SPECS = [
    ("FAENet", "runs/qata_paper0516_qata_faenet_visual_clean_seed42", (211, 104, 73)),
    ("CXR-TGFS", "runs/screening0506_qata_cxr_frozen_keep_both_seed42", (214, 161, 72)),
    ("Ours", "runs/qata_paper0516_qata_simple_native_zero_both_seed42", (64, 128, 108)),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    if sys.platform.startswith("win"):
        candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        ]
    else:
        candidates = []
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(30, True)
FONT_HEADER = font(18, True)
FONT_LABEL = font(15, True)
FONT_SMALL = font(13, False)


def load_metric_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "mask_name": row["mask_name"],
                    "dice": float(row["dice"]),
                    "iou": float(row["iou"]),
                    "target_pixels": float(row["target_pixels"]),
                    "pred_pixels": float(row["pred_pixels"]),
                }
            )
    return rows


def build_dataset_index(ds: QaTaCOV19Dataset) -> dict[str, int]:
    index = {}
    for i in range(len(ds)):
        sample = ds[i]
        index[str(sample["mask_name"])] = i
    return index


def select_rows(rows: list[dict], mode: str, num_cases: int) -> list[dict]:
    positive = [r for r in rows if r["target_pixels"] > 10]
    if mode == "best":
        # Avoid only giant diffuse cases: require a usable but not extreme lesion area.
        usable = [r for r in positive if 450 <= r["target_pixels"] <= 18000] or positive
        return sorted(usable, key=lambda r: r["dice"], reverse=True)[:num_cases]

    # Mixed paper figure: representative high quality, small lesion, large lesion,
    # median case, and one hard/failure case.
    target_pixels = np.array([r["target_pixels"] for r in positive], dtype=np.float32)
    dice_values = np.array([r["dice"] for r in positive], dtype=np.float32)
    q25, q50, q75 = np.quantile(target_pixels, [0.25, 0.5, 0.75])
    median_dice = float(np.median(dice_values))
    picks: list[dict] = []

    def add(candidates: list[dict], key, reverse: bool = True) -> None:
        used = {p["mask_name"] for p in picks}
        for item in sorted(candidates, key=key, reverse=reverse):
            if item["mask_name"] not in used:
                picks.append(item)
                return

    add([r for r in positive if q25 <= r["target_pixels"] <= q75], key=lambda r: r["dice"], reverse=True)
    add([r for r in positive if r["target_pixels"] <= q25], key=lambda r: r["dice"], reverse=True)
    add([r for r in positive if r["target_pixels"] >= q75], key=lambda r: r["dice"], reverse=True)
    add(positive, key=lambda r: -abs(r["dice"] - median_dice), reverse=True)
    add([r for r in positive if r["target_pixels"] >= q25], key=lambda r: r["dice"], reverse=False)

    for row in sorted(positive, key=lambda r: r["dice"], reverse=True):
        if len(picks) >= num_cases:
            break
        if row["mask_name"] not in {p["mask_name"] for p in picks}:
            picks.append(row)
    return picks[:num_cases]


def resize_rgb(arr: np.ndarray, tile: int) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB").resize((tile, tile), Image.BILINEAR)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.46) -> np.ndarray:
    out = base.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    if base is None:
        canvas = np.ones((*gt.shape, 3), dtype=np.uint8) * 245
    else:
        canvas = (base.astype(np.float32) * 0.62 + 255.0 * 0.38).clip(0, 255).astype(np.uint8)
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = pred & gt
    fp = pred & ~gt
    fn = (~pred) & gt
    canvas[tp] = np.array([52, 150, 112], dtype=np.uint8)
    canvas[fp] = np.array([66, 133, 190], dtype=np.uint8)
    canvas[fn] = np.array([220, 74, 67], dtype=np.uint8)
    return canvas


def draw_cell(
    out: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: np.ndarray,
    x: int,
    y: int,
    tile: int,
    label: str | None = None,
    score: str | None = None,
) -> None:
    draw.rounded_rectangle((x - 4, y - 4, x + tile + 4, y + tile + 4), radius=8, fill=(248, 249, 247), outline=(221, 225, 226), width=1)
    out.paste(resize_rgb(img, tile), (x, y))
    if score:
        box_w = 70
        draw.rounded_rectangle((x + tile - box_w - 7, y + 7, x + tile - 7, y + 30), radius=7, fill=(255, 255, 255), outline=(210, 214, 216), width=1)
        draw.text((x + tile - box_w, y + 10), score, fill=(33, 37, 41), font=FONT_SMALL)
    if label:
        draw.text((x + 4, y + tile + 9), label, fill=(38, 42, 48), font=FONT_SMALL)


def render_figure(
    ds: QaTaCOV19Dataset,
    selected: list[dict],
    predictions: dict[str, dict[str, dict]],
    out_path: Path,
    summary_path: Path,
    tile: int,
    title: str,
) -> None:
    cols = ["Input", "GT", *(name for name, _, _ in RUN_SPECS), "Error (Ours)"]
    left_w = 72
    margin = 32
    gap = 13
    top_h = 104
    row_gap = 34
    row_h = tile + row_gap
    width = margin * 2 + left_w + len(cols) * tile + (len(cols) - 1) * gap
    height = top_h + len(selected) * row_h + 54
    out = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    draw.text((margin, 24), title, fill=(22, 25, 30), font=FONT_TITLE)
    draw.text(
        (margin, 62),
        "Clean qualitative grid: prompts and filenames are stored in JSON, not printed inside the figure.",
        fill=(91, 101, 116),
        font=FONT_SMALL,
    )

    x0 = margin + left_w
    for ci, col in enumerate(cols):
        x = x0 + ci * (tile + gap)
        draw.text((x + 4, top_h - 27), col, fill=(32, 37, 43), font=FONT_HEADER)

    summary = []
    for ri, row in enumerate(selected, start=1):
        y = top_h + (ri - 1) * row_h
        sample_idx = row["dataset_index"]
        sample = ds[sample_idx]
        image = sample["image"].squeeze(0).numpy()
        gt = sample["mask"].squeeze(0).numpy() > 0.5
        base = gray_to_rgb(image)

        draw.text((margin, y + tile // 2 - 12), f"C{ri}", fill=(35, 39, 46), font=FONT_LABEL)

        x = x0
        draw_cell(out, draw, base, x, y, tile)
        x += tile + gap
        draw_cell(out, draw, overlay_mask(base, gt, (52, 150, 112), alpha=0.55), x, y, tile, score=f"{gt.mean()*100:.1f}%")
        x += tile + gap

        case_scores: dict[str, float] = {}
        for name, _, color in RUN_SPECS:
            pred = predictions[name][row["mask_name"]]["pred"] > 0.5
            dice = predictions[name][row["mask_name"]]["dice"]
            case_scores[name] = dice
            draw_cell(out, draw, overlay_mask(base, pred, color, alpha=0.50), x, y, tile, score=f"D {dice:.2f}")
            x += tile + gap

        ours_pred = predictions["Ours"][row["mask_name"]]["pred"] > 0.5
        draw_cell(out, draw, error_map(ours_pred, gt, base=base), x, y, tile)

        summary.append(
            {
                "case_id": ri,
                "dataset_index": sample_idx,
                "mask_name": row["mask_name"],
                "prompt": str(sample["text"]),
                "selector_dice": row["dice"],
                "selector_iou": row["iou"],
                "target_pixels": row["target_pixels"],
                "gt_area_ratio": float(gt.mean()),
                **{f"dice_{k}": v for k, v in case_scores.items()},
            }
        )

    legend_y = height - 34
    legend_items = [("TP", (52, 150, 112)), ("FP", (66, 133, 190)), ("FN", (220, 74, 67))]
    lx = margin + left_w
    draw.text((margin, legend_y - 1), "Error map:", fill=(91, 101, 116), font=FONT_SMALL)
    for label, color in legend_items:
        draw.rounded_rectangle((lx, legend_y, lx + 18, legend_y + 18), radius=4, fill=color)
        draw.text((lx + 24, legend_y), label, fill=(91, 101, 116), font=FONT_SMALL)
        lx += 72

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    # PDF wrapper keeps image visual identical and easy to include in LaTeX.
    out.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Create clean paper-style QaTa qualitative figures.")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--metric-csv", type=str, default="runs/qata_paper0516_qata_simple_native_zero_both_seed42/test_per_image_metrics.csv")
    parser.add_argument("--mode", choices=["mixed", "best"], default="mixed")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit QaTa test dataset indices. If set, overrides --mode selection.",
    )
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--tile", type=int, default=154)
    parser.add_argument("--out-dir", type=str, default="generated_figures/qata_qualitative_clean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = QaTaCOV19Dataset(root_dir=args.data_root, split="test", image_size=224, use_text=True)
    index_by_mask = build_dataset_index(ds)

    metric_rows = load_metric_rows(ROOT / args.metric_csv)
    metrics_by_name = {r["mask_name"]: r for r in metric_rows}
    selected = []
    if args.sample_ids:
        for sample_id in args.sample_ids:
            sample = ds[sample_id]
            mask_name = str(sample["mask_name"])
            metric_row = dict(
                metrics_by_name.get(
                    mask_name,
                    {
                        "mask_name": mask_name,
                        "dice": float("nan"),
                        "iou": float("nan"),
                        "target_pixels": float(sample["mask"].sum().item()),
                        "pred_pixels": float("nan"),
                    },
                )
            )
            metric_row["dataset_index"] = sample_id
            selected.append(metric_row)
    else:
        for row in select_rows(metric_rows, args.mode, args.num_cases):
            if row["mask_name"] not in index_by_mask:
                continue
            row = dict(row)
            row["dataset_index"] = index_by_mask[row["mask_name"]]
            selected.append(row)
    if not selected:
        raise RuntimeError("No selected rows matched the QaTa test dataset.")

    predictions: dict[str, dict[str, dict]] = {}
    for run_name, run_dir, _color in RUN_SPECS:
        print(f"Loading {run_name}: {run_dir}", flush=True)
        model, tokenizer, cfg, threshold = load_run(ROOT / run_dir, device)
        run_preds = {}
        for row in selected:
            sample = ds[row["dataset_index"]]
            gt = sample["mask"].squeeze(0).numpy()
            result = predict_one(model, tokenizer, cfg, sample["image"], str(sample["text"]), device, threshold=threshold)
            dice = sample_dice(result["pred"], gt)
            run_preds[row["mask_name"]] = {
                "pred": result["pred"],
                "dice": float(dice),
                "threshold": float(threshold),
            }
        predictions[run_name] = run_preds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = ROOT / args.out_dir
    suffix = "selected" if args.sample_ids else args.mode
    render_figure(
        ds=ds,
        selected=selected,
        predictions=predictions,
        out_path=out_dir / f"fig_qata_qualitative_clean_{suffix}.png",
        summary_path=out_dir / f"fig_qata_qualitative_clean_{suffix}_cases.json",
        tile=args.tile,
        title=(
            "QaTa-COV19 selected qualitative examples"
            if args.sample_ids
            else (
                "QaTa-COV19 qualitative segmentation examples"
                if args.mode == "mixed"
                else "QaTa-COV19 high-performing examples"
            )
        ),
    )
    print(f"Wrote clean qualitative figure set to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
