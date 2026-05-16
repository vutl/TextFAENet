from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import QaTaCOV19Dataset
from scripts.qata_vis_utils import (
    FONT_LABEL,
    FONT_SMALL,
    FONT_SUBTITLE,
    FONT_TITLE,
    PALETTE,
    load_run,
    predict_one,
)


STAGES = ["dec4", "dec3", "dec2", "dec1"]
BANDS = ["LL", "LH", "HL", "HH"]
BAND_KEYS = {"LL": "a_ll_mean", "LH": "a_lh_mean", "HL": "a_hl_mean", "HH": "a_hh_mean"}
BAND_COLORS = {
    "LL": (58, 111, 183),
    "LH": (37, 146, 113),
    "HL": (220, 145, 52),
    "HH": (126, 96, 157),
}


def collect_gate_stats(
    ds: QaTaCOV19Dataset,
    run_dir: Path,
    device: torch.device,
    max_samples: int,
) -> list[dict]:
    model, tokenizer, cfg, threshold = load_run(run_dir, device)
    rows: list[dict] = []
    with torch.no_grad():
        for idx in range(min(max_samples, len(ds))):
            sample = ds[idx]
            gt = sample["mask"].squeeze(0).numpy()
            area = float(gt.mean())
            if area <= 0.0:
                continue
            result = predict_one(
                model,
                tokenizer,
                cfg,
                sample["image"],
                str(sample["text"]),
                device,
                threshold=threshold,
                capture_debug=True,
            )
            debug = result["debug"]
            for stage in STAGES:
                stage_debug = debug.get(stage)
                if not stage_debug:
                    continue
                row = {
                    "index": idx,
                    "mask_name": sample["mask_name"],
                    "area": area,
                    "stage": stage,
                }
                for band in BANDS:
                    row[f"a_{band}"] = float(stage_debug[BAND_KEYS[band]][0].item())
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    areas = np.array([r["area"] for r in rows if r["stage"] == "dec1"], dtype=np.float32)
    q25, q75 = np.quantile(areas, [0.25, 0.75])
    for row in rows:
        if row["area"] <= q25:
            row["size_group"] = "small"
        elif row["area"] >= q75:
            row["size_group"] = "large"
        else:
            row["size_group"] = "mid"

    summary: list[dict] = []
    for group in ["all", "small", "large"]:
        for stage in STAGES:
            subset = [r for r in rows if r["stage"] == stage and (group == "all" or r["size_group"] == group)]
            if not subset:
                continue
            out = {
                "group": group,
                "stage": stage,
                "count": len(subset),
            }
            for band in BANDS:
                values = [float(r[f"a_{band}"]) for r in subset]
                out[f"mean_a_{band}"] = float(np.mean(values))
                out[f"std_a_{band}"] = float(np.std(values))
            summary.append(out)
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_group_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    summary: list[dict],
    group: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=(248, 250, 252), outline=(224, 228, 235), width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=PALETTE["text"], font=FONT_LABEL)
    chart_left = x0 + 76
    chart_right = x1 - 32
    chart_top = y0 + 76
    chart_bottom = y1 - 72
    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill=(170, 178, 190), width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(170, 178, 190), width=2)
    for val in [0.3, 0.4, 0.5, 0.6, 0.7]:
        frac = (val - 0.3) / 0.4
        y = chart_bottom - int(frac * (chart_bottom - chart_top))
        draw.line([(chart_left, y), (chart_right, y)], fill=(228, 232, 238), width=1)
        draw.text((x0 + 24, y - 8), f"{val:.1f}", fill=PALETTE["muted"], font=FONT_SMALL)

    by_stage = {r["stage"]: r for r in summary if r["group"] == group}
    slot = (chart_right - chart_left) / len(STAGES)
    bar_w = 22
    for stage_idx, stage in enumerate(STAGES):
        cx = int(chart_left + stage_idx * slot + slot / 2)
        row = by_stage.get(stage)
        if not row:
            continue
        offsets = [-39, -13, 13, 39]
        for band, off in zip(BANDS, offsets):
            value = float(row[f"mean_a_{band}"])
            frac = max(0.0, min(1.0, (value - 0.3) / 0.4))
            y = chart_bottom - int(frac * (chart_bottom - chart_top))
            draw.rounded_rectangle((cx + off - bar_w // 2, y, cx + off + bar_w // 2, chart_bottom), radius=5, fill=BAND_COLORS[band])
        draw.text((cx - 20, chart_bottom + 16), stage, fill=PALETTE["text"], font=FONT_SMALL)


def make_figure(summary: list[dict], out_path: Path) -> None:
    w, h = 1650, 950
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((58, 34), "TGFS Frequency Gate Statistics on QaTa", fill=PALETTE["text"], font=FONT_TITLE)
    draw.text(
        (58, 78),
        "Mean text-guided sub-band gates across decoder stages, split by GT lesion area.",
        fill=PALETTE["muted"],
        font=FONT_SUBTITLE,
    )

    boxes = [
        (58, 142, 806, 590),
        (846, 142, 1594, 590),
    ]
    draw_group_panel(draw, boxes[0], "Small lesions: bottom area quartile", summary, "small")
    draw_group_panel(draw, boxes[1], "Large lesions: top area quartile", summary, "large")

    legend_x = 58
    legend_y = 660
    draw.text((legend_x, legend_y), "Legend", fill=PALETTE["text"], font=FONT_LABEL)
    for idx, band in enumerate(BANDS):
        x = legend_x + idx * 145
        draw.rounded_rectangle((x, legend_y + 38, x + 34, legend_y + 60), radius=5, fill=BAND_COLORS[band])
        draw.text((x + 44, legend_y + 37), f"alpha_{band}", fill=PALETTE["text"], font=FONT_SMALL)
    draw.text(
        (58, 760),
        "Interpretation note: these are learned gate activations, not direct energy measurements. "
        "Use them to show how text-conditioned frequency selection changes by stage and lesion scale.",
        fill=PALETTE["muted"],
        font=FONT_SMALL,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser("Dump and plot QaTa TGFS gate statistics.")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--run-dir", type=str, default="runs/qata_diag0516_qata_simple_native_keep_both_seed42")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="generated_figures/qata_qualitative/gate_stats")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=224, use_text=True)
    out_dir = Path(args.out_dir)

    rows = collect_gate_stats(ds, Path(args.run_dir), device, args.max_samples)
    summary = summarize(rows)
    save_csv(rows, out_dir / "gate_stats_raw.csv")
    save_csv(summary, out_dir / "gate_stats_summary.csv")
    (out_dir / "gate_stats_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(summary, out_dir / "fig_qata_gate_stats_small_vs_large.png")
    print(f"Saved gate statistics to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
