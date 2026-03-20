from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Subset
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import LFAENetTGFS, LFAENetTGFSv2


def build_model_from_cfg(cfg: dict, device: torch.device):
    common_kwargs = {
        "in_channels": 1,
        "num_classes": 1,
        "text_dim": cfg.get("text_dim", 256),
        "text_encoder_type": "biomedvlp-cxr-bert" if cfg.get("use_cxr_bert", True) else "simple",
        "text_backbone_path": cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
        "freeze_text_backbone": cfg.get("freeze_text_backbone", True),
    }
    model_type = cfg.get("model_type", "lfaenet_tgfs")
    if model_type == "lfaenet_tgfs_v2":
        model = LFAENetTGFSv2(
            **common_kwargs,
            drop_hh_in_decoder=cfg.get("drop_hh_in_decoder", True),
        )
    else:
        model = LFAENetTGFS(**common_kwargs)
    return model.to(device)


def to_rgb_gray(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    g = (x * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def make_triplet_panel(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, text: str) -> Image.Image:
    base = to_rgb_gray(img)
    gt_overlay = overlay_mask(base, gt > 0.5, (0, 255, 0), alpha=0.45)
    pred_overlay = overlay_mask(base, pred > 0.5, (255, 0, 0), alpha=0.45)

    h, w, _ = base.shape
    header_h = 44
    canvas = np.ones((header_h + h, w * 3, 3), dtype=np.uint8) * 255
    canvas[header_h:, :w] = base
    canvas[header_h:, w : 2 * w] = gt_overlay
    canvas[header_h:, 2 * w :] = pred_overlay

    out = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), "Input", fill=(10, 10, 10))
    draw.text((w + 8, 8), "GT overlay", fill=(10, 10, 10))
    draw.text((2 * w + 8, 8), "Pred overlay", fill=(10, 10, 10))
    draw.text((8, 24), text[:120], fill=(70, 70, 70))
    return out


def predict_one(model, tokenizer, sample, device, max_text_len: int):
    img = sample["image"].unsqueeze(0).to(device)
    text = sample["text"]
    toks = tokenizer([text], padding="max_length", truncation=True, max_length=max_text_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            img,
            token_ids=toks["input_ids"].to(device),
            attention_mask=toks["attention_mask"].to(device),
        )
        pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].detach().cpu().numpy()

    return pred


def main() -> None:
    parser = argparse.ArgumentParser("Inference visualization for train/test/val")
    parser.add_argument("--run-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--num-train", type=int, default=2)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen/infer_vis")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
        trust_remote_code=True,
        local_files_only=True,
    )

    image_size = cfg.get("image_size", 224)
    train_ds = QaTaCOV19Dataset(root_dir=args.data_root, split="train", image_size=image_size)
    test_ds = QaTaCOV19Dataset(root_dir=args.data_root, split="test", image_size=image_size)

    # No official val split in QaTa-COV19-v2 here; use a held-out tail from train as val proxy.
    val_count = min(max(args.num_val * 20, 100), len(train_ds) // 5)
    val_start = len(train_ds) - val_count
    val_indices = list(range(val_start, len(train_ds)))
    val_ds = Subset(train_ds, val_indices)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def export_samples(ds, n: int, split_name: str):
        for i in range(min(n, len(ds))):
            sample = ds[i]
            pred = predict_one(model, tokenizer, sample, device, cfg.get("max_text_len", 64))
            img = sample["image"].squeeze(0).numpy()
            gt = sample["mask"].squeeze(0).numpy()
            panel = make_triplet_panel(img, gt, pred, sample["text"])
            panel.save(out_dir / f"{split_name}_{i:03d}_panel.png")

            Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(out_dir / f"{split_name}_{i:03d}_pred.png")
            Image.fromarray((gt * 255).astype(np.uint8), mode="L").save(out_dir / f"{split_name}_{i:03d}_gt.png")

    export_samples(train_ds, args.num_train, "train")
    export_samples(test_ds, args.num_test, "test")
    export_samples(val_ds, args.num_val, "val_proxy")

    (out_dir / "README.txt").write_text(
        "val_proxy is a held-out tail subset from Train because dataset does not provide a separate val split here.\n",
        encoding="utf-8",
    )

    print(f"Saved inference visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
