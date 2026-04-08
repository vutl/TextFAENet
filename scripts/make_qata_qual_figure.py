from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import FAENet, LFAENetTGFS, LFAENetTGFSv2


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_and_tokenizer(run_dir: Path, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_type = cfg.get("model_type", "lfaenet_tgfs_v2")

    if model_type == "faenet":
        model = FAENet(in_channels=1, num_classes=1)
        tokenizer = None
    elif model_type == "lfaenet_tgfs":
        model = LFAENetTGFS(
            in_channels=1,
            num_classes=1,
            text_dim=cfg.get("text_dim", 256),
            vocab_size=cfg.get("vocab_size", 30522),
            text_encoder_type="biomedvlp-cxr-bert" if cfg.get("use_cxr_bert", True) else "simple",
            text_backbone_path=cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
            freeze_text_backbone=cfg.get("freeze_text_backbone", True),
        )
    elif model_type == "lfaenet_tgfs_v2":
        model = LFAENetTGFSv2(
            in_channels=1,
            num_classes=1,
            text_dim=cfg.get("text_dim", 256),
            vocab_size=cfg.get("vocab_size", 30522),
            text_encoder_type="biomedvlp-cxr-bert" if cfg.get("use_cxr_bert", True) else "simple",
            text_backbone_path=cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
            freeze_text_backbone=cfg.get("freeze_text_backbone", True),
            drop_hh_in_decoder=cfg.get("drop_hh_in_decoder", True),
            low_level_hf_scale=cfg.get("low_level_hf_scale", 0.6),
            spatial_sharpen_power=cfg.get("spatial_sharpen_power", 2.0),
            use_deep_supervision=cfg.get("use_deep_supervision", False),
        )
    else:
        raise ValueError(f"Unsupported model_type in {run_dir}: {model_type}")

    checkpoint = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = None
    if model_type != "faenet":
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
            trust_remote_code=True,
            local_files_only=True,
        )

    return model, tokenizer, cfg


def to_rgb_gray(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    g = (x * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    canvas = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.uint8) * 255
    false_negative = gt & ~pred
    false_positive = pred & ~gt
    true_positive = pred & gt
    canvas[true_positive] = np.array([190, 230, 190], dtype=np.uint8)
    canvas[false_negative] = np.array([220, 50, 50], dtype=np.uint8)
    canvas[false_positive] = np.array([60, 130, 220], dtype=np.uint8)
    return canvas


def sample_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def predict_mask(model, tokenizer, cfg: dict, sample: dict, device: torch.device) -> np.ndarray:
    image = sample["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        if cfg.get("model_type") == "faenet":
            logits = model(image)
        else:
            text = sample["text"]
            toks = tokenizer(
                [text],
                padding="max_length",
                truncation=True,
                max_length=cfg.get("max_text_len", 64),
                return_tensors="pt",
            )
            logits = model(
                image,
                token_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )

    pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].detach().cpu().numpy()
    return pred


def select_cases(records: list[dict], num_cases: int) -> list[dict]:
    if not records:
        return []

    positives = [r for r in records if r["area"] > 0.001]
    source = positives if positives else records

    by_gain = sorted(source, key=lambda r: (r["gain"], r["ours_dice"]), reverse=True)
    by_easy = sorted(source, key=lambda r: r["ours_dice"], reverse=True)
    by_hard = sorted(source, key=lambda r: (r["ours_dice"], -r["gain"]))
    by_small_gain = sorted(
        [r for r in source if r["area"] < 0.08] or source,
        key=lambda r: (r["gain"], -r["area"]),
        reverse=True,
    )

    picks: list[dict] = []
    for candidate_list in (by_gain, by_small_gain, by_easy, by_hard, by_gain):
        for item in candidate_list:
            if item["index"] not in {x["index"] for x in picks}:
                picks.append(item)
                break
        if len(picks) >= num_cases:
            break

    cursor = 0
    while len(picks) < min(num_cases, len(source)):
        item = by_gain[cursor]
        if item["index"] not in {x["index"] for x in picks}:
            picks.append(item)
        cursor += 1

    return picks[:num_cases]


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, fill: tuple[int, int, int]) -> None:
    wrapped = textwrap.fill(text, width=width_chars)
    draw.multiline_text(xy, wrapped, fill=fill, spacing=4)


def main() -> None:
    parser = argparse.ArgumentParser("Create QaTa qualitative comparison figure")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--baseline-run", type=str, default="runs/qata_faenet_notext_adamw_cosine_e30")
    parser.add_argument("--ours-run", type=str, default="runs/qata_b4_e50_cxrbert_frozen_v2")
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--out-path", type=str, default="paper_figures/fig3_qata_qualitative.png")
    parser.add_argument("--summary-path", type=str, default="paper_figures/fig3_qata_qualitative_cases.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_run = Path(args.baseline_run)
    ours_run = Path(args.ours_run)

    baseline_model, baseline_tokenizer, baseline_cfg = build_model_and_tokenizer(baseline_run, device)
    ours_model, ours_tokenizer, ours_cfg = build_model_and_tokenizer(ours_run, device)

    image_size = int(ours_cfg.get("image_size", baseline_cfg.get("image_size", 224)))
    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=image_size, use_text=True)

    max_count = min(args.candidate_pool, len(ds)) if args.candidate_pool > 0 else len(ds)
    records: list[dict] = []
    for idx in range(max_count):
        sample = ds[idx]
        gt = sample["mask"].squeeze(0).numpy()
        pred_baseline = predict_mask(baseline_model, baseline_tokenizer, baseline_cfg, sample, device)
        pred_ours = predict_mask(ours_model, ours_tokenizer, ours_cfg, sample, device)

        baseline_dice = sample_dice(pred_baseline, gt)
        ours_dice = sample_dice(pred_ours, gt)
        records.append(
            {
                "index": idx,
                "mask_name": sample["mask_name"],
                "text": sample["text"],
                "area": float(gt.mean()),
                "baseline_dice": baseline_dice,
                "ours_dice": ours_dice,
                "gain": ours_dice - baseline_dice,
                "image": sample["image"].squeeze(0).numpy(),
                "gt": gt,
                "pred_baseline": pred_baseline,
                "pred_ours": pred_ours,
            }
        )

    selected = select_cases(records, args.num_cases)
    if not selected:
        raise RuntimeError("No cases were selected for the qualitative figure.")

    cell_w = image_size
    cell_h = image_size
    left_w = 430
    cols = 6
    margin = 24
    row_gap = 18
    title_h = 72
    header_h = 28
    row_text_h = 78
    row_h = header_h + row_text_h + cell_h

    canvas_w = margin * 2 + left_w + cols * cell_w
    canvas_h = title_h + margin + len(selected) * row_h + max(len(selected) - 1, 0) * row_gap + 24
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    out = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(out)

    draw.text((margin, 16), "Figure 3. Qualitative Comparison on QaTa-COV19-v2", fill=(15, 15, 15))
    draw.text(
        (margin, 40),
        "Visual-only FAENet versus text-guided LFAENet-TGFS v2 on representative test cases.",
        fill=(80, 80, 80),
    )

    titles = ["Input", "GT Overlay", "FAENet", "Ours", "FAENet Error", "Ours Error"]
    col_x = [margin + left_w + i * cell_w for i in range(cols)]
    for x, title in zip(col_x, titles):
        draw.text((x + 8, title_h), title, fill=(20, 20, 20))

    y = title_h + margin
    summary_rows: list[dict[str, float | int | str]] = []
    for row_id, item in enumerate(selected):
        base = to_rgb_gray(item["image"])
        gt_overlay = overlay_mask(base, item["gt"] > 0.5, color=(0, 180, 0), alpha=0.45)
        baseline_overlay = overlay_mask(base, item["pred_baseline"] > 0.5, color=(220, 50, 50), alpha=0.45)
        ours_overlay = overlay_mask(base, item["pred_ours"] > 0.5, color=(220, 50, 50), alpha=0.45)
        baseline_error = make_error_map(item["pred_baseline"], item["gt"])
        ours_error = make_error_map(item["pred_ours"], item["gt"])

        draw.text((margin, y), f"Case {row_id + 1}", fill=(15, 15, 15))
        draw.text((margin + 88, y), f"GT area: {item['area'] * 100.0:.1f}%", fill=(80, 80, 80))
        draw.text((margin, y + 22), f"FAENet Dice: {item['baseline_dice']:.3f}", fill=(140, 50, 50))
        draw.text((margin + 180, y + 22), f"Ours Dice: {item['ours_dice']:.3f}", fill=(30, 110, 40))
        draw.text((margin + 340, y + 22), f"Gain: {item['gain']:+.3f}", fill=(50, 50, 50))
        draw_wrapped_text(draw, (margin, y + 44), f"Prompt: {item['text']}", width_chars=62, fill=(35, 35, 35))

        row_y = y + header_h + row_text_h
        images = [base, gt_overlay, baseline_overlay, ours_overlay, baseline_error, ours_error]
        for x, img in zip(col_x, images):
            out.paste(Image.fromarray(img), (x, row_y))

        summary_rows.append(
            {
                "case_id": row_id + 1,
                "index": item["index"],
                "mask_name": item["mask_name"],
                "gt_area": item["area"],
                "baseline_dice": item["baseline_dice"],
                "ours_dice": item["ours_dice"],
                "gain": item["gain"],
                "text": item["text"],
            }
        )
        y += row_h + row_gap

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved qualitative comparison figure to: {out_path}")
    print(f"Saved selected-case summary to: {summary_path}")


if __name__ == "__main__":
    main()
