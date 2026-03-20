from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import LFAENetTGFSv2


def build_model_from_cfg(cfg: dict, device: torch.device) -> LFAENetTGFSv2:
    model = LFAENetTGFSv2(
        in_channels=1,
        num_classes=1,
        text_dim=cfg.get("text_dim", 256),
        text_encoder_type="biomedvlp-cxr-bert" if cfg.get("use_cxr_bert", True) else "simple",
        text_backbone_path=cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
        freeze_text_backbone=cfg.get("freeze_text_backbone", True),
        drop_hh_in_decoder=cfg.get("drop_hh_in_decoder", True),
    ).to(device)
    return model


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    den = arr.max() - arr.min()
    if den < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr / den
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def overlay_heatmap(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    mask = mask.astype(np.float32)[..., None]
    out = (1.0 - alpha * mask) * base + (alpha * mask) * red
    return out.clip(0, 255).astype(np.uint8)


def export_mask_panel(path: Path, input_gray: np.ndarray, spatial_mask: np.ndarray, stage_name: str) -> None:
    heat = to_uint8(spatial_mask)
    over = overlay_heatmap(input_gray, heat.astype(np.float32) / 255.0)

    h, w = input_gray.shape
    header_h = 24
    canvas = np.ones((header_h + h, w * 3, 3), dtype=np.uint8) * 255
    canvas[header_h:, :w] = np.stack([input_gray, input_gray, input_gray], axis=-1)
    canvas[header_h:, w : 2 * w] = np.stack([heat, heat, heat], axis=-1)
    canvas[header_h:, 2 * w :] = over

    out = Image.fromarray(canvas, mode="RGB")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(out)
    draw.text((8, 5), "Input", fill=(10, 10, 10))
    draw.text((w + 8, 5), f"{stage_name} spatial", fill=(10, 10, 10))
    draw.text((2 * w + 8, 5), "Overlay", fill=(10, 10, 10))
    out.save(path)


def main() -> None:
    parser = argparse.ArgumentParser("Export TGFS v2 spatial masks and frequency gate stats")
    parser.add_argument("--run-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen/tgfs_debug")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "lfaenet_tgfs_v2":
        raise ValueError("This debug exporter currently supports model_type=lfaenet_tgfs_v2 only.")

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    model.set_debug_capture(True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
        trust_remote_code=True,
        local_files_only=True,
    )

    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=cfg.get("image_size", 224))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "spatial_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []

    with torch.no_grad():
        for idx in range(min(args.num_samples, len(ds))):
            sample = ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            text = sample["text"]
            toks = tokenizer([text], padding="max_length", truncation=True, max_length=cfg.get("max_text_len", 64), return_tensors="pt")

            logits = model(
                image,
                token_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )
            _ = torch.sigmoid(logits)

            debug = model.get_debug_outputs()
            input_gray = to_uint8(sample["image"].squeeze(0).numpy())

            for stage_name in ("dec4", "dec3", "dec2", "dec1"):
                stage_debug = debug.get(stage_name)
                if stage_debug is None:
                    continue

                spatial = stage_debug["spatial_mask"][0:1]
                spatial = F.interpolate(spatial, size=input_gray.shape, mode="bilinear", align_corners=False)
                spatial_np = spatial.squeeze().numpy()

                export_mask_panel(
                    path=mask_dir / f"{args.split}_{idx:03d}_{stage_name}_spatial.png",
                    input_gray=input_gray,
                    spatial_mask=spatial_np,
                    stage_name=stage_name,
                )

                rows.append(
                    {
                        "split": args.split,
                        "index": idx,
                        "mask_name": sample["mask_name"],
                        "stage": stage_name,
                        "a_LL": float(stage_debug["a_ll_mean"][0].item()),
                        "a_LH": float(stage_debug["a_lh_mean"][0].item()),
                        "a_HL": float(stage_debug["a_hl_mean"][0].item()),
                        "a_HH": float(stage_debug["a_hh_mean"][0].item()),
                    }
                )

    csv_path = out_dir / "gate_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "index", "mask_name", "stage", "a_LL", "a_LH", "a_HL", "a_HH"])
        writer.writeheader()
        writer.writerows(rows)

    # Per-stage summary for quick reading.
    summary_rows: list[dict[str, str | float]] = []
    for stage_name in ("dec4", "dec3", "dec2", "dec1"):
        stage_data = [r for r in rows if r["stage"] == stage_name]
        if not stage_data:
            continue
        summary_rows.append(
            {
                "stage": stage_name,
                "count": len(stage_data),
                "mean_a_LL": float(np.mean([r["a_LL"] for r in stage_data])),
                "mean_a_LH": float(np.mean([r["a_LH"] for r in stage_data])),
                "mean_a_HL": float(np.mean([r["a_HL"] for r in stage_data])),
                "mean_a_HH": float(np.mean([r["a_HH"] for r in stage_data])),
            }
        )

    summary_path = out_dir / "gate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "count", "mean_a_LL", "mean_a_LH", "mean_a_HL", "mean_a_HH"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved TGFS debug outputs to: {out_dir}")


if __name__ == "__main__":
    main()
