from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import LFAENetTGFS, LFAENetTGFSv2
from src.models.lfaenet_tgfs_v2 import haar_dwt2d


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


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    den = arr.max() - arr.min()
    if den < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr / den
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    im = Image.fromarray(arr, mode="L")
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(im)


def save_panel(path: Path, images: list[np.ndarray], titles: list[str]) -> None:
    h, w = images[0].shape
    header_h = 28
    canvas = np.ones((header_h + h, w * len(images)), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        x0 = i * w
        canvas[header_h : header_h + h, x0 : x0 + w] = img

    out = Image.fromarray(canvas, mode="L").convert("RGB")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(out)
    for i, t in enumerate(titles):
        draw.text((6 + i * w, 6), t, fill=(10, 10, 10))

    out.save(path)


def main() -> None:
    parser = argparse.ArgumentParser("Frequency mode check (LL/LH/HL/HH)")
    parser.add_argument("--run-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="runs/qata_b4_e50_cxrbert_frozen/freq_check")
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

    ds = QaTaCOV19Dataset(root_dir=args.data_root, split="test", image_size=cfg.get("image_size", 224))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(min(args.num_samples, len(ds))):
            sample = ds[idx]
            x = sample["image"].unsqueeze(0).to(device)

            # Reuse model native DWT when available (v1), fallback to functional DWT (v2).
            if hasattr(model.enc1[1], "dwt"):
                ll, lh, hl, hh = model.enc1[1].dwt(x)
            else:
                ll, lh, hl, hh = haar_dwt2d(x)

            x_img = to_uint8(sample["image"].squeeze(0).numpy())
            ll_img = to_uint8(ll[0, 0].detach().cpu().numpy())
            lh_img = to_uint8(np.abs(lh[0, 0].detach().cpu().numpy()))
            hl_img = to_uint8(np.abs(hl[0, 0].detach().cpu().numpy()))
            hh_img = to_uint8(np.abs(hh[0, 0].detach().cpu().numpy()))

            h, w = x_img.shape
            ll_img = resize_to(ll_img, h, w)
            lh_img = resize_to(lh_img, h, w)
            hl_img = resize_to(hl_img, h, w)
            hh_img = resize_to(hh_img, h, w)

            panel_path = out_dir / f"test_{idx:03d}_freq_panel.png"
            save_panel(panel_path, [x_img, ll_img, lh_img, hl_img, hh_img], ["input", "LL", "LH", "HL", "HH"])

            text = sample["text"]
            toks = tokenizer([text], padding="max_length", truncation=True, max_length=cfg.get("max_text_len", 64), return_tensors="pt")
            logits = model(
                x,
                token_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )
            pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].detach().cpu().numpy()
            pred_img = (pred * 255).astype(np.uint8)
            Image.fromarray(pred_img, mode="L").save(out_dir / f"test_{idx:03d}_pred_mask.png")

    print(f"Saved frequency checks to: {out_dir}")


if __name__ == "__main__":
    main()
