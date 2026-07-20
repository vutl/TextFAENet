from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qata_vis_utils import load_run, predict_one, sample_dice  # noqa: E402
from scripts.train_brain_tumors import CsvPromptedFolderSegmentationDataset  # noqa: E402
from src.data.mosmed_text_csv import MosMedTextCSVDataset  # noqa: E402
from src.data.qata_cov19 import QaTaCOV19Dataset  # noqa: E402


QATA_RUN = ROOT / "runs" / "qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42"
QATA_DATA = Path(r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
QATA_CASES = ROOT / "generated_figures" / "paper_qualitative_panels" / "qata_picked_high_gap" / "fig_qata_external_qualitative_cases.json"
PROMPT_DIR = ROOT / "generated_figures" / "paper_qualitative_panels" / "brain_breast_prompt_protocol"
MOSMED_RUN = ROOT / "runs" / "mosmed_v9e_448"
MOSMED_DATA = ROOT / "datasets" / "MosMed"

COLORS = {
    "pred": (48, 143, 113),
    "gt": (45, 170, 130),
    "fp": (53, 126, 189),
    "fn": (226, 75, 74),
    "text": (31, 36, 45),
    "muted": (90, 102, 120),
}


def font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def normalize(x) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.squeeze().detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    m = mask.astype(bool)
    rgb[m] = 0.42 * rgb[m] + 0.58 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def error(gray: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    p, g = pred.astype(bool), gt.astype(bool)
    for mask, color in [(p & g, COLORS["pred"]), (p & ~g, COLORS["fp"]), (~p & g, COLORS["fn"])]:
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def load_qata(device: torch.device) -> dict:
    case = json.loads(QATA_CASES.read_text(encoding="utf-8"))[4]
    ds = QaTaCOV19Dataset(root_dir=QATA_DATA, split="test", image_size=224)
    sample = ds[int(case["dataset_index"])]
    model, tokenizer, cfg, threshold = load_run(QATA_RUN, device)
    result = predict_one(model, tokenizer, cfg, sample["image"], str(sample["text"]), device, threshold=threshold)
    gt = sample["mask"].squeeze().numpy().astype(np.float32)
    return {
        "dataset": "QaTa-COV19",
        "name": str(sample["mask_name"]),
        "image": normalize(sample["image"]),
        "gt": gt,
        "pred": result["pred"],
        "dice": sample_dice(result["pred"], gt),
    }


def load_mosmed(device: torch.device) -> dict:
    ds = MosMedTextCSVDataset(root_dir=str(MOSMED_DATA), split="test", image_size=448)
    model, tokenizer, cfg, threshold = load_run(MOSMED_RUN, device)
    candidate_ids = sorted(set(int(x) for x in np.linspace(0, len(ds) - 1, 24)))
    records = []
    with torch.no_grad():
        for idx in candidate_ids:
            sample = ds[idx]
            gt = sample["mask"].squeeze().numpy().astype(np.float32)
            if float(gt.mean()) < 0.001:
                continue
            result = predict_one(model, tokenizer, cfg, sample["image"], str(sample["text"]), device, threshold=threshold)
            records.append((sample_dice(result["pred"], gt), idx, sample, gt, result["pred"]))
    if not records:
        raise RuntimeError("No valid MosMed candidates were found for the qualitative panel.")
    dice, idx, sample, gt, pred = max(records, key=lambda item: item[0])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset": "MosMedData+",
        "name": str(sample["mask_name"]),
        "index": idx,
        "selection": "highest Dice among 24 evenly spaced test candidates with foreground area >= 0.1%",
        "image": normalize(sample["image"]),
        "gt": gt,
        "pred": pred,
        "dice": dice,
    }


def load_tumor(dataset_name: str, display_name: str) -> dict:
    cases = json.loads((PROMPT_DIR / "fig_brain_breast_prompt_protocol_cases.json").read_text(encoding="utf-8"))
    case = next(x for x in cases if x["dataset"] == dataset_name)
    root = ROOT / "datasets" / ("brain_tumors" if dataset_name == "brain" else "breast_tumors")
    ds = CsvPromptedFolderSegmentationDataset(root_dir=root, split="test", image_size=320)
    sample = next(ds[i] for i in range(len(ds)) if str(ds[i]["mask_name"]) == case["name"])
    pred_path = PROMPT_DIR / "pred_masks" / dataset_name / "structured" / case["name"]
    pred = (np.asarray(Image.open(pred_path).convert("L")) > 127).astype(np.float32)
    gt = sample["mask"].squeeze().numpy().astype(np.float32)
    return {
        "dataset": display_name,
        "name": case["name"],
        "image": normalize(sample["image"]),
        "gt": gt,
        "pred": pred,
        "dice": sample_dice(pred, gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Make cross-dataset qualitative panel.")
    parser.add_argument("--tile", type=int, default=230)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "generated_figures" / "paper_qualitative_panels" / "cross_dataset_qata_brain_breast",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [load_qata(device), load_mosmed(device), load_tumor("brain", "Brain MRI"), load_tumor("breast", "Breast US")]

    columns = ["Input", "Ground truth", "Prediction", "Error map"]
    tile, gap, margin, label_w, header_h = args.tile, 14, 28, 125, 52
    width = margin * 2 + label_w + len(columns) * tile + (len(columns) - 1) * gap
    height = margin + header_h + len(rows) * tile + (len(rows) - 1) * gap + 45
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x0 = margin + label_w
    for col, label in enumerate(columns):
        draw.text((x0 + col * (tile + gap) + 4, margin + 7), label, fill=COLORS["text"], font=font(18, True))

    records = []
    for idx, row in enumerate(rows):
        y = margin + header_h + idx * (tile + gap)
        draw.text((margin, y + 8), row["dataset"], fill=COLORS["text"], font=font(16, True))
        panels = [
            Image.fromarray(row["image"]).convert("RGB"),
            overlay(row["image"], row["gt"], COLORS["gt"]),
            overlay(row["image"], row["pred"], COLORS["pred"]),
            error(row["image"], row["pred"], row["gt"]),
        ]
        for col, panel in enumerate(panels):
            x = x0 + col * (tile + gap)
            canvas.paste(panel.resize((tile, tile), Image.Resampling.BILINEAR), (x, y))
            if col == 2:
                badge = f"D {row['dice']:.2f}"
                draw.rounded_rectangle((x + tile - 60, y + 7, x + tile - 7, y + 31), radius=6, fill="white")
                draw.text((x + tile - 54, y + 10), badge, fill=COLORS["text"], font=font(13))
        records.append({k: row[k] for k in ["dataset", "name", "dice", "index", "selection"] if k in row})

    ly = height - 28
    draw.text((margin, ly), "Error:", fill=COLORS["muted"], font=font(12))
    lx = margin + 48
    for label, color in [("TP", COLORS["pred"]), ("FP", COLORS["fp"]), ("FN", COLORS["fn"])]:
        draw.rounded_rectangle((lx, ly, lx + 14, ly + 14), radius=3, fill=color)
        draw.text((lx + 19, ly - 1), label, fill=COLORS["muted"], font=font(12))
        lx += 58

    out_png = out_dir / "fig_cross_dataset_qata_mosmed_brain_breast.png"
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=300)
    (out_dir / "fig_cross_dataset_qata_mosmed_brain_breast_cases.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
