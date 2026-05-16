from __future__ import annotations

import argparse
import json
import os
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
    draw_tile,
    draw_wrapped_text,
    error_map,
    gray_to_rgb,
    load_run,
    mask_rgb,
    overlay_mask,
    predict_one,
    sample_dice,
)


DEFAULT_RUNS = [
    ("CXR native", "runs/screening0506_qata_cxr_frozen_keep_both_seed42"),
    ("CXR empty", "runs/qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42"),
    ("CXR shuffle", "runs/qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42"),
    ("Simple native", "runs/qata_diag0516_qata_simple_native_keep_both_seed42"),
]


def parse_run_specs(specs: list[str] | None) -> list[tuple[str, Path]]:
    if not specs:
        return [(name, ROOT / path) for name, path in DEFAULT_RUNS]
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be name=run_dir, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), Path(path.strip())))
    return parsed


def select_cases(
    ds: QaTaCOV19Dataset,
    selector_run: Path,
    device: torch.device,
    candidate_pool: int,
    num_cases: int,
) -> list[dict]:
    model, tokenizer, cfg, threshold = load_run(selector_run, device)
    records: list[dict] = []
    with torch.no_grad():
        for idx in range(min(candidate_pool, len(ds))):
            sample = ds[idx]
            image = sample["image"]
            gt = sample["mask"].squeeze(0).numpy()
            result = predict_one(model, tokenizer, cfg, image, str(sample["text"]), device, threshold=threshold)
            dice = sample_dice(result["pred"], gt)
            area = float(gt.mean())
            if area <= 0.0:
                continue
            records.append(
                {
                    "index": idx,
                    "mask_name": sample["mask_name"],
                    "area": area,
                    "dice": dice,
                    "text": sample["text"],
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not records:
        raise RuntimeError("No positive-mask candidate cases found.")

    areas = np.array([r["area"] for r in records], dtype=np.float32)
    q25, q50, q75 = np.quantile(areas, [0.25, 0.50, 0.75])

    picks: list[dict] = []

    def add_best(candidates: list[dict], key, reverse: bool = True) -> None:
        for item in sorted(candidates, key=key, reverse=reverse):
            if item["index"] not in {p["index"] for p in picks}:
                picks.append(item)
                return

    # Easy representative, large lesion, small lesion, median case, and a failure/hard case.
    add_best([r for r in records if r["area"] >= q50], key=lambda r: r["dice"], reverse=True)
    add_best([r for r in records if r["area"] >= q75], key=lambda r: r["dice"], reverse=True)
    add_best([r for r in records if r["area"] <= q25], key=lambda r: r["dice"], reverse=True)
    add_best(records, key=lambda r: -abs(r["dice"] - float(np.median([x["dice"] for x in records]))), reverse=True)
    add_best([r for r in records if r["area"] >= 0.003], key=lambda r: r["dice"], reverse=False)

    for item in sorted(records, key=lambda r: r["dice"], reverse=True):
        if len(picks) >= num_cases:
            break
        if item["index"] not in {p["index"] for p in picks}:
            picks.append(item)

    return picks[:num_cases]


def make_figure(
    ds: QaTaCOV19Dataset,
    selected: list[dict],
    run_specs: list[tuple[str, Path]],
    out_path: Path,
    summary_path: Path,
    device: torch.device,
    tile: int = 168,
) -> None:
    # Compute predictions one run at a time to keep CXR-BERT memory bounded.
    pred_bank: dict[str, dict[int, dict]] = {}
    for run_name, run_dir in run_specs:
        print(f"Loading run for qualitative grid: {run_name} ({run_dir})", flush=True)
        model, tokenizer, cfg, threshold = load_run(run_dir, device)
        run_preds: dict[int, dict] = {}
        for item in selected:
            sample = ds[item["index"]]
            image = sample["image"]
            gt = sample["mask"].squeeze(0).numpy()
            result = predict_one(model, tokenizer, cfg, image, str(sample["text"]), device, threshold=threshold)
            dice = sample_dice(result["pred"], gt)
            run_preds[item["index"]] = {
                "pred": result["pred"],
                "prob": result["prob"],
                "dice": dice,
                "threshold": threshold,
            }
        pred_bank[run_name] = run_preds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cols = 2 + len(run_specs) + 1
    left_w = 360
    margin = 42
    gap = 16
    title_h = 112
    row_h = tile + 92
    w = margin * 2 + left_w + cols * tile + (cols - 1) * gap
    h = title_h + len(selected) * row_h + 44
    out = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(out)

    draw.text((margin, 32), "QaTa Qualitative Segmentation Results", fill=PALETTE["text"], font=FONT_TITLE)
    draw.text(
        (margin, 76),
        "Input, ground truth, predictions from trained ablation checkpoints, and final error map.",
        fill=PALETTE["muted"],
        font=FONT_SUBTITLE,
    )

    summary_rows: list[dict] = []
    y = title_h
    for row_id, item in enumerate(selected, start=1):
        sample = ds[item["index"]]
        image = sample["image"].squeeze(0).numpy()
        gt = sample["mask"].squeeze(0).numpy()
        base = gray_to_rgb(image)

        draw.text((margin, y + 4), f"Case {row_id}", fill=PALETTE["text"], font=FONT_LABEL)
        draw.text((margin + 82, y + 5), f"idx={item['index']}  area={item['area'] * 100:.2f}%", fill=PALETTE["muted"], font=FONT_SMALL)
        draw_wrapped_text(draw, (margin, y + 34), f"Prompt: {str(sample['text'])}", width_chars=48, fill=PALETTE["text"])

        x = margin + left_w
        draw_tile(out, draw, base, (x, y + 46), "Input", tile=tile)
        x += tile + gap
        draw_tile(out, draw, overlay_mask(base, gt > 0.5, color=PALETTE["green"]), (x, y + 46), "GT overlay", f"area {gt.mean() * 100:.1f}%", tile=tile)
        x += tile + gap

        run_dices: dict[str, float] = {}
        for run_name, _ in run_specs:
            pred = pred_bank[run_name][item["index"]]["pred"]
            dice = pred_bank[run_name][item["index"]]["dice"]
            run_dices[run_name] = dice
            pred_overlay = overlay_mask(base, pred > 0.5, color=PALETTE["red"])
            draw_tile(out, draw, pred_overlay, (x, y + 46), run_name, f"Dice {dice:.3f}", tile=tile)
            x += tile + gap

        final_name = run_specs[-1][0]
        err = error_map(pred_bank[final_name][item["index"]]["pred"], gt)
        draw_tile(out, draw, err, (x, y + 46), f"Error: {final_name}", "green TP | blue FP | red FN", tile=tile)

        summary_rows.append(
            {
                "case_id": row_id,
                "index": item["index"],
                "mask_name": item["mask_name"],
                "gt_area": item["area"],
                "selector_dice": item["dice"],
                "text": str(sample["text"]),
                **{f"dice_{name}": dice for name, dice in run_dices.items()},
            }
        )
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved qualitative grid to: {out_path}", flush=True)
    print(f"Saved case summary to: {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("Make QaTa qualitative segmentation grid from existing checkpoints.")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--selector-run", type=str, default="runs/qata_diag0516_qata_simple_native_keep_both_seed42")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional list of name=run_dir entries.")
    parser.add_argument("--candidate-pool", type=int, default=120)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None)
    parser.add_argument("--out-path", type=str, default="generated_figures/qata_qualitative/fig_qata_qualitative_segmentation.png")
    parser.add_argument("--summary-path", type=str, default="generated_figures/qata_qualitative/fig_qata_qualitative_cases.json")
    parser.add_argument("--tile", type=int, default=168)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=224, use_text=True)
    run_specs = parse_run_specs(args.runs)

    if args.sample_ids:
        selected = []
        for idx in args.sample_ids:
            sample = ds[idx]
            gt = sample["mask"].squeeze(0).numpy()
            selected.append(
                {
                    "index": idx,
                    "mask_name": sample["mask_name"],
                    "area": float(gt.mean()),
                    "dice": float("nan"),
                    "text": sample["text"],
                }
            )
    else:
        selected = select_cases(ds, Path(args.selector_run), device, args.candidate_pool, args.num_cases)

    make_figure(
        ds=ds,
        selected=selected,
        run_specs=run_specs,
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        device=device,
        tile=args.tile,
    )


if __name__ == "__main__":
    main()
