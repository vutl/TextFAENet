from __future__ import annotations

import argparse
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
    draw_tile,
    draw_wrapped_text,
    gray_to_rgb,
    heatmap_red,
    load_run,
    overlay_mask,
    predict_one,
    resize_rgb,
    sample_dice,
    spatial_mask_from_debug,
)


def prompt_variants(ds: QaTaCOV19Dataset, idx: int) -> dict[str, str]:
    native = str(ds[idx]["text"])
    shuffle_idx = (idx + 37) % len(ds)
    if shuffle_idx == idx:
        shuffle_idx = (idx + 1) % len(ds)
    return {
        "native": native,
        "empty": "",
        "shuffle": str(ds[shuffle_idx]["text"]),
        "generic": "segment the abnormal medical region.",
    }


def score_cases(
    ds: QaTaCOV19Dataset,
    run_dir: Path,
    device: torch.device,
    candidate_pool: int,
    num_cases: int,
    selection_mode: str,
) -> list[dict]:
    model, tokenizer, cfg, threshold = load_run(run_dir, device)
    records: list[dict] = []
    with torch.no_grad():
        for idx in range(min(candidate_pool, len(ds))):
            sample = ds[idx]
            gt = sample["mask"].squeeze(0).numpy()
            area = float(gt.mean())
            if area <= 0.0:
                continue
            image = sample["image"]
            prompts = prompt_variants(ds, idx)
            native = predict_one(model, tokenizer, cfg, image, prompts["native"], device, threshold=threshold)
            empty = predict_one(model, tokenizer, cfg, image, prompts["empty"], device, threshold=threshold)
            shuffle = predict_one(model, tokenizer, cfg, image, prompts["shuffle"], device, threshold=threshold)
            native_dice = sample_dice(native["pred"], gt)
            empty_dice = sample_dice(empty["pred"], gt)
            shuffle_dice = sample_dice(shuffle["pred"], gt)
            records.append(
                {
                    "index": idx,
                    "mask_name": sample["mask_name"],
                    "area": area,
                    "native_dice": native_dice,
                    "empty_dice": empty_dice,
                    "shuffle_dice": shuffle_dice,
                    "min_delta": native_dice - max(empty_dice, shuffle_dice),
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not records:
        raise RuntimeError("No positive-mask candidate cases found.")

    if selection_mode == "strongest":
        eligible = [r for r in records if r["native_dice"] >= 0.5 and r["area"] >= 0.001]
        return sorted(eligible, key=lambda r: (r["min_delta"], r["native_dice"]), reverse=True)[:num_cases]

    areas = np.array([r["area"] for r in records], dtype=np.float32)
    q25, q50, q75 = np.quantile(areas, [0.25, 0.50, 0.75])
    picks: list[dict] = []

    def add(candidates: list[dict], key, reverse: bool = True) -> None:
        for item in sorted(candidates, key=key, reverse=reverse):
            if item["index"] not in {p["index"] for p in picks}:
                picks.append(item)
                return

    add([r for r in records if r["area"] >= q50], key=lambda r: r["min_delta"], reverse=True)
    add([r for r in records if r["area"] <= q25], key=lambda r: r["min_delta"], reverse=True)
    add([r for r in records if r["area"] >= q75], key=lambda r: r["native_dice"], reverse=True)
    add(records, key=lambda r: -abs(r["native_dice"] - float(np.median([x["native_dice"] for x in records]))), reverse=True)
    add([r for r in records if r["area"] >= 0.003], key=lambda r: r["native_dice"], reverse=False)

    for item in sorted(records, key=lambda r: r["min_delta"], reverse=True):
        if len(picks) >= num_cases:
            break
        if item["index"] not in {p["index"] for p in picks}:
            picks.append(item)

    return picks[:num_cases]


def make_figure(
    ds: QaTaCOV19Dataset,
    selected: list[dict],
    run_dir: Path,
    out_path: Path,
    summary_path: Path,
    device: torch.device,
    stage: str,
    tile: int = 158,
) -> None:
    model, tokenizer, cfg, threshold = load_run(run_dir, device)
    run_label = run_dir.name
    variants = ["native", "empty", "shuffle"]

    cols = 2 + len(variants) + len(variants)
    left_w = 360
    margin = 42
    gap = 14
    title_h = 118
    row_h = tile + 96
    w = margin * 2 + left_w + cols * tile + (cols - 1) * gap
    h = title_h + len(selected) * row_h + 46
    out = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(out)

    draw.text((margin, 30), "Prompt Intervention: Same Checkpoint, Different Text", fill=PALETTE["text"], font=FONT_TITLE)
    draw.text(
        (margin, 75),
        f"Checkpoint fixed: {run_label}. Predictions use threshold={threshold:.2f}; spatial masks use {stage}.",
        fill=PALETTE["muted"],
        font=FONT_SUBTITLE,
    )

    summary_rows: list[dict] = []
    y = title_h
    with torch.no_grad():
        for row_id, item in enumerate(selected, start=1):
            sample = ds[item["index"]]
            image = sample["image"]
            image_np = image.squeeze(0).numpy()
            gt = sample["mask"].squeeze(0).numpy()
            base = gray_to_rgb(image_np)
            prompts = prompt_variants(ds, item["index"])

            draw.text((margin, y + 4), f"Case {row_id}", fill=PALETTE["text"], font=FONT_LABEL)
            draw.text((margin + 82, y + 5), f"idx={item['index']}  area={gt.mean() * 100:.2f}%", fill=PALETTE["muted"], font=FONT_SMALL)
            draw_wrapped_text(draw, (margin, y + 34), f"Native prompt: {str(sample['text'])}", width_chars=48, fill=PALETTE["text"])

            x = margin + left_w
            draw_tile(out, draw, base, (x, y + 50), "Input", tile=tile)
            x += tile + gap
            draw_tile(out, draw, overlay_mask(base, gt > 0.5, color=PALETTE["green"]), (x, y + 50), "GT overlay", f"area {gt.mean() * 100:.1f}%", tile=tile)
            x += tile + gap

            pred_results: dict[str, dict] = {}
            for name in variants:
                result = predict_one(
                    model,
                    tokenizer,
                    cfg,
                    image,
                    prompts[name],
                    device,
                    threshold=threshold,
                    capture_debug=True,
                )
                dice = sample_dice(result["pred"], gt)
                pred_results[name] = {**result, "dice": dice}
                pred_overlay = overlay_mask(base, result["pred"] > 0.5, color=PALETTE["red"])
                draw_tile(out, draw, pred_overlay, (x, y + 50), f"{name} pred", f"Dice {dice:.3f}", tile=tile)
                x += tile + gap

            for name in variants:
                spatial = spatial_mask_from_debug(pred_results[name]["debug"], image_shape=gt.shape, stage=stage)
                if spatial is None:
                    ms_img = np.ones_like(base) * 245
                else:
                    ms_img = heatmap_red(spatial, base=base, alpha=0.55)
                draw_tile(out, draw, ms_img, (x, y + 50), f"{name} M_s", stage, tile=tile)
                x += tile + gap

            summary_rows.append(
                {
                    "case_id": row_id,
                    "index": item["index"],
                    "mask_name": sample["mask_name"],
                    "gt_area": float(gt.mean()),
                    "native_dice": pred_results["native"]["dice"],
                    "empty_dice": pred_results["empty"]["dice"],
                    "shuffle_dice": pred_results["shuffle"]["dice"],
                    "native_prompt": prompts["native"],
                    "shuffle_prompt": prompts["shuffle"],
                }
            )
            y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    out.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved prompt intervention figure to: {out_path}", flush=True)
    print(f"Saved case summary to: {summary_path}", flush=True)


def make_compact_figure(
    ds: QaTaCOV19Dataset,
    selected: list[dict],
    run_dir: Path,
    out_path: Path,
    summary_path: Path,
    device: torch.device,
    stage: str,
    tile: int = 170,
) -> None:
    model, tokenizer, cfg, threshold = load_run(run_dir, device)
    variants = ["native", "empty", "shuffle"]
    headers = ["Input", "Ground truth", "Native", "Empty", "Shuffled", "Native M_s", "Empty M_s", "Shuffled M_s"]
    margin_x, margin_y, gap, row_gap, label_w = 34, 34, 12, 42, 34
    header_h, metric_h = 38, 24
    width = margin_x * 2 + label_w + len(headers) * tile + (len(headers) - 1) * gap
    height = margin_y * 2 + header_h + len(selected) * (tile + metric_h) + (len(selected) - 1) * row_gap
    out = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(out)

    start_x = margin_x + label_w
    for col, header in enumerate(headers):
        x = start_x + col * (tile + gap)
        bbox = draw.textbbox((0, 0), header, font=FONT_LABEL)
        draw.text((x + (tile - (bbox[2] - bbox[0])) / 2, margin_y), header, fill=PALETTE["text"], font=FONT_LABEL)

    summary_rows: list[dict] = []
    with torch.no_grad():
        for row_id, item in enumerate(selected):
            y = margin_y + header_h + row_id * (tile + metric_h + row_gap)
            sample = ds[item["index"]]
            image = sample["image"]
            image_np = image.squeeze(0).numpy()
            gt = sample["mask"].squeeze(0).numpy()
            base = gray_to_rgb(image_np)
            prompts = prompt_variants(ds, item["index"])
            draw.text((margin_x + 4, y + tile // 2 - 10), chr(ord("A") + row_id), fill=PALETTE["text"], font=FONT_LABEL)

            images = [base, overlay_mask(base, gt > 0.5, color=PALETTE["green"])]
            pred_results: dict[str, dict] = {}
            for name in variants:
                result = predict_one(model, tokenizer, cfg, image, prompts[name], device, threshold=threshold, capture_debug=True)
                dice = sample_dice(result["pred"], gt)
                pred_results[name] = {**result, "dice": dice}
                images.append(overlay_mask(base, result["pred"] > 0.5, color=PALETTE["green"]))

            for name in variants:
                spatial = spatial_mask_from_debug(pred_results[name]["debug"], image_shape=gt.shape, stage=stage)
                images.append(np.ones_like(base) * 245 if spatial is None else heatmap_red(spatial, base=base, alpha=0.55))

            for col, img in enumerate(images):
                x = start_x + col * (tile + gap)
                out.paste(resize_rgb(img, tile), (x, y))
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(205, 211, 220), width=1)
                if 2 <= col <= 4:
                    name = variants[col - 2]
                    label = f"Dice {pred_results[name]['dice']:.3f}"
                    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
                    draw.text((x + (tile - (bbox[2] - bbox[0])) / 2, y + tile + 4), label, fill=PALETTE["muted"], font=FONT_SMALL)

            summary_rows.append(
                {
                    "case_id": chr(ord("A") + row_id),
                    "index": item["index"],
                    "mask_name": sample["mask_name"],
                    "native_dice": pred_results["native"]["dice"],
                    "empty_dice": pred_results["empty"]["dice"],
                    "shuffle_dice": pred_results["shuffle"]["dice"],
                    "native_prompt": prompts["native"],
                    "shuffle_prompt": prompts["shuffle"],
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    out.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved compact prompt intervention figure to: {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("Make same-checkpoint QaTa prompt intervention visualization.")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--run-dir", type=str, default="runs/qata_diag0516_qata_simple_native_keep_both_seed42")
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--selection-mode", choices=["diverse", "strongest"], default="diverse")
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None)
    parser.add_argument("--stage", type=str, choices=["dec4", "dec3", "dec2", "dec1"], default="dec1")
    parser.add_argument("--out-path", type=str, default="generated_figures/qata_qualitative/fig_qata_prompt_intervention.png")
    parser.add_argument("--summary-path", type=str, default="generated_figures/qata_qualitative/fig_qata_prompt_intervention_cases.json")
    parser.add_argument("--tile", type=int, default=158)
    parser.add_argument("--compact", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=224, use_text=True)

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
                }
            )
    else:
        selected = score_cases(
            ds,
            Path(args.run_dir),
            device,
            args.candidate_pool,
            args.num_cases,
            args.selection_mode,
        )

    renderer = make_compact_figure if args.compact else make_figure
    renderer(
        ds=ds,
        selected=selected,
        run_dir=Path(args.run_dir),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        device=device,
        stage=args.stage,
        tile=args.tile,
    )


if __name__ == "__main__":
    main()
