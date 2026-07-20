from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_brain_breast_per_image import (
    ROOT,
    TextSegCollator,
    build_test_loader,
    checkpoint_path,
    create_model,
    load_checkpoint,
    logits_with_optional_tta,
    namespace_from_config,
    read_json,
)


RUNS = {
    "brain": {
        "structured": ROOT / "runs" / "paper0623_brain_structured_v3resnet50cxr_both_seed42",
        "medclip": ROOT / "runs" / "paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42",
    },
    "breast": {
        "structured": ROOT / "runs" / "paper0623_breast_structured_v3resnet50cxr_both_seed42",
        "medclip": ROOT / "runs" / "paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42",
    },
}

COLORS = {
    "structured": (48, 143, 113),
    "medclip": (142, 103, 178),
    "gt": (45, 170, 130),
    "fp": (53, 126, 189),
    "fn": (226, 75, 74),
    "text": (31, 36, 45),
    "muted": (90, 102, 120),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def normalize_image(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze().detach().cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def error_overlay(gray: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    for mask, color in [
        (pred_b & gt_b, COLORS["structured"]),
        (pred_b & ~gt_b, COLORS["fp"]),
        (~pred_b & gt_b, COLORS["fn"]),
    ]:
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred * gt).sum())
    return (2 * inter + 1e-6) / (float(pred.sum() + gt.sum()) + 1e-6)


def select_cases(dataset: str, count: int) -> list[str]:
    structured = pd.read_csv(RUNS[dataset]["structured"] / "test_per_image_metrics.csv")
    medclip = pd.read_csv(RUNS[dataset]["medclip"] / "test_per_image_metrics.csv")
    merged = structured.merge(medclip, on="name", suffixes=("_structured", "_medclip"))
    merged["delta"] = merged["dice_structured"] - merged["dice_medclip"]
    # Avoid zero-output outliers; select clear but visually credible prompt effects.
    eligible = merged[
        (merged["dice_structured"] >= 0.80)
        & (merged["dice_medclip"] >= 0.35)
        & (merged["gt_area_ratio_structured"] >= 0.003)
    ].sort_values(["delta", "dice_structured"], ascending=False)
    if len(eligible) < count:
        raise RuntimeError(f"Not enough eligible {dataset} cases")
    return eligible.head(count)["name"].astype(str).tolist()


@torch.inference_mode()
def infer_run(run_dir: Path, names: list[str], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    cfg = read_json(run_dir / "config.json")
    args = namespace_from_config(cfg, use_amp=True)
    model, tokenizer = create_model(args, device)
    ckpt = load_checkpoint(checkpoint_path(run_dir), device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    final = read_json(run_dir / "final_test.json")
    threshold = float(final.get("best_threshold", ckpt.get("best_threshold", 0.5)))
    dataset, _ = build_test_loader(args, tokenizer, batch_size=1, num_workers=0, device=device)
    by_name = {str(dataset[i]["mask_name"]): i for i in range(len(dataset))}
    collator = TextSegCollator(
        tokenizer=tokenizer,
        max_length=int(args.max_text_len),
        prompt_source=str(getattr(args, "prompt_source", "csv")),
        fixed_prompt=str(getattr(args, "fixed_prompt", "Segment the tumor region.")),
    )
    preds: dict[str, np.ndarray] = {}
    samples: dict[str, dict] = {}
    for name in names:
        sample = dataset[by_name[name]]
        batch = collator([sample])
        targets, logits = logits_with_optional_tta(batch, model, args, device, tokenizer)
        pred = (torch.sigmoid(logits.float())[0, 0] > threshold).float().cpu().numpy()
        preds[name] = pred
        samples[name] = sample
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, samples


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def main() -> None:
    parser = argparse.ArgumentParser("Make Brain/Breast prompt-protocol qualitative panel.")
    parser.add_argument("--cases-per-dataset", type=int, default=2)
    parser.add_argument("--tile", type=int, default=190)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "generated_figures" / "paper_qualitative_panels" / "brain_breast_prompt_protocol",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = {dataset: select_cases(dataset, args.cases_per_dataset) for dataset in RUNS}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {d: {} for d in RUNS}
    samples: dict[str, dict[str, dict]] = {d: {} for d in RUNS}
    for dataset in ["brain", "breast"]:
        for protocol in ["medclip", "structured"]:
            print(f"Inferring {dataset} {protocol}...", flush=True)
            pred, current_samples = infer_run(RUNS[dataset][protocol], selected[dataset], device)
            predictions[dataset][protocol] = pred
            samples[dataset].update(current_samples)

    tile, gap, margin = args.tile, 12, 30
    columns = ["Input", "GT", "MedCLIP-style", "Structured", "Error (structured)"]
    rows = sum(len(v) for v in selected.values())
    header_h, row_label_w = 54, 72
    width = margin * 2 + row_label_w + len(columns) * tile + (len(columns) - 1) * gap
    height = margin + header_h + rows * tile + (rows - 1) * gap + 46
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x0 = margin + row_label_w
    for col, label in enumerate(columns):
        draw.text((x0 + col * (tile + gap) + 5, margin + 8), label, fill=COLORS["text"], font=font(17, True))

    records = []
    row = 0
    for dataset in ["brain", "breast"]:
        for local_idx, name in enumerate(selected[dataset], start=1):
            sample = samples[dataset][name]
            gray = normalize_image(sample["image"])
            gt = sample["mask"].squeeze().detach().cpu().numpy().astype(np.float32)
            pred_m = predictions[dataset]["medclip"][name]
            pred_s = predictions[dataset]["structured"][name]
            panels = [
                Image.fromarray(gray).convert("RGB"),
                overlay(gray, gt, COLORS["gt"]),
                overlay(gray, pred_m, COLORS["medclip"]),
                overlay(gray, pred_s, COLORS["structured"]),
                error_overlay(gray, pred_s, gt),
            ]
            y = margin + header_h + row * (tile + gap)
            draw.text((margin, y + 7), dataset.title(), fill=COLORS["text"], font=font(15, True))
            draw.text((margin, y + 28), f"#{local_idx}", fill=COLORS["muted"], font=font(13))
            for col, panel in enumerate(panels):
                panel = panel.resize((tile, tile), Image.Resampling.BILINEAR)
                x = x0 + col * (tile + gap)
                canvas.paste(panel, (x, y))
                if col in {2, 3}:
                    score = dice(pred_m if col == 2 else pred_s, gt)
                    badge = f"D {score:.2f}"
                    box = draw.textbbox((0, 0), badge, font=font(13))
                    bw = box[2] - box[0] + 14
                    draw.rounded_rectangle((x + tile - bw - 6, y + 6, x + tile - 6, y + 30), radius=6, fill="white")
                    draw.text((x + tile - bw + 1, y + 9), badge, fill=COLORS["text"], font=font(13))
            save_mask(pred_s, out_dir / "pred_masks" / dataset / "structured" / name)
            save_mask(pred_m, out_dir / "pred_masks" / dataset / "medclip" / name)
            records.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "structured_dice": dice(pred_s, gt),
                    "medclip_style_dice": dice(pred_m, gt),
                    "structured_prompt": str(samples[dataset][name]["text"]),
                }
            )
            row += 1

    ly = height - 29
    draw.text((margin, ly), "Error:", fill=COLORS["muted"], font=font(12))
    lx = margin + 48
    for label, color in [("TP", COLORS["structured"]), ("FP", COLORS["fp"]), ("FN", COLORS["fn"])]:
        draw.rounded_rectangle((lx, ly, lx + 14, ly + 14), radius=3, fill=color)
        draw.text((lx + 19, ly - 1), label, fill=COLORS["muted"], font=font(12))
        lx += 58

    out_png = out_dir / "fig_brain_breast_prompt_protocol.png"
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=300)
    (out_dir / "fig_brain_breast_prompt_protocol_cases.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
