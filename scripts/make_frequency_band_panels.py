from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.mosmed_text_csv import MosMedTextCSVDataset
from src.data.prompted_folder_seg import PromptedFolderSegmentationDataset
from src.data.qata_cov19 import QaTaCOV19Dataset
from src.modules.wavelet import HaarDWT2D


PALETTE = {
    "bg": (250, 248, 242),
    "panel": (255, 255, 255),
    "ink": (31, 36, 42),
    "muted": (92, 100, 112),
    "accent": (40, 114, 113),
    "gt": (52, 168, 83),
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(24, True)
FONT_LABEL = font(16, True)
FONT_SMALL = font(12, False)


def to_uint8_gray(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 100.0 - percentile)
    hi = np.percentile(arr, percentile)
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo)
    return (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def signed_band_rgb(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Blue/orange signed visualization with magnitude encoded by saturation."""
    arr = arr.astype(np.float32)
    scale = float(np.percentile(np.abs(arr), percentile))
    if scale < 1e-8:
        return np.ones((*arr.shape, 3), dtype=np.uint8) * 245
    x = np.clip(arr / scale, -1.0, 1.0)
    mag = np.abs(x)[..., None]
    base = np.ones((*arr.shape, 3), dtype=np.float32) * 245.0
    pos = np.array([222.0, 112.0, 58.0], dtype=np.float32)
    neg = np.array([48.0, 103.0, 180.0], dtype=np.float32)
    color = np.where(x[..., None] >= 0.0, pos, neg)
    rgb = base * (1.0 - mag) + color * mag
    return rgb.clip(0, 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    gray = to_uint8_gray(image)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    m = mask > 0.5
    color = np.array(PALETTE["gt"], dtype=np.float32)
    rgb[m] = rgb[m] * (1.0 - alpha) + color * alpha
    return rgb.clip(0, 255).astype(np.uint8)


def resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((size, size), resample=Image.BILINEAR))


def resize_gray(arr: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((size, size), resample=Image.BILINEAR))


def band_energies(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> dict[str, float]:
    vals = {
        "LL": float(np.mean(np.abs(ll))),
        "LH": float(np.mean(np.abs(lh))),
        "HL": float(np.mean(np.abs(hl))),
        "HH": float(np.mean(np.abs(hh))),
    }
    total = sum(vals.values()) + 1e-12
    return {k: v / total for k, v in vals.items()}


def draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    arr: np.ndarray,
    x: int,
    y: int,
    tile: int,
    title: str,
    subtitle: str = "",
) -> None:
    img = Image.fromarray(arr).resize((tile, tile), resample=Image.BILINEAR)
    canvas.paste(img, (x, y))
    draw.rounded_rectangle((x, y, x + tile, y + tile), radius=8, outline=(218, 213, 203), width=1)
    draw.text((x, y + tile + 8), title, fill=PALETTE["ink"], font=FONT_LABEL)
    if subtitle:
        draw.text((x, y + tile + 29), subtitle, fill=PALETTE["muted"], font=FONT_SMALL)


def make_panel(
    sample: dict[str, Any],
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    energies: dict[str, float],
    out_path: Path,
) -> None:
    image = sample["image"].squeeze(0).numpy()
    mask = sample["mask"].squeeze(0).numpy()
    mask_name = str(sample["mask_name"])

    tile = 178
    gap = 18
    margin = 34
    header = 76
    footer = 112
    cols = 6
    width = margin * 2 + cols * tile + (cols - 1) * gap
    height = header + tile + footer
    canvas = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 22), "Wavelet frequency decomposition", fill=PALETTE["ink"], font=FONT_TITLE)
    short_name = mask_name if len(mask_name) <= 68 else mask_name[:65] + "..."
    draw.text((margin, 51), short_name, fill=PALETTE["muted"], font=FONT_SMALL)

    tiles = [
        (
            np.stack([to_uint8_gray(image)] * 3, axis=-1),
            "Input",
            "image",
        ),
        (
            overlay_mask(image, mask),
            "GT overlay",
            f"lesion {mask.mean() * 100:.2f}%",
        ),
        (
            np.stack([to_uint8_gray(ll)] * 3, axis=-1),
            "LL",
            f"structure {energies['LL'] * 100:.1f}%",
        ),
        (
            signed_band_rgb(lh),
            "LH",
            f"h-edge {energies['LH'] * 100:.1f}%",
        ),
        (
            signed_band_rgb(hl),
            "HL",
            f"v-edge {energies['HL'] * 100:.1f}%",
        ),
        (
            signed_band_rgb(hh),
            "HH",
            f"texture {energies['HH'] * 100:.1f}%",
        ),
    ]
    y = header
    for i, (arr, title, subtitle) in enumerate(tiles):
        x = margin + i * (tile + gap)
        draw_tile(canvas, draw, arr, x, y, tile, title, subtitle)

    legend_y = header + tile + 62
    draw.rectangle((margin, legend_y, margin + 18, legend_y + 12), fill=(222, 112, 58))
    draw.text((margin + 24, legend_y - 3), "positive coefficient", fill=PALETTE["muted"], font=FONT_SMALL)
    draw.rectangle((margin + 160, legend_y, margin + 178, legend_y + 12), fill=(48, 103, 180))
    draw.text((margin + 184, legend_y - 3), "negative coefficient", fill=PALETTE["muted"], font=FONT_SMALL)
    draw.text(
        (margin + 360, legend_y - 3),
        "Band percentages are mean absolute coefficient shares.",
        fill=PALETTE["muted"],
        font=FONT_SMALL,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_dataset(args: argparse.Namespace):
    if args.dataset == "qata":
        return QaTaCOV19Dataset(args.data_root, split=args.split, image_size=args.image_size, use_text=True)
    if args.dataset in {"brain", "breast", "prompted"}:
        return PromptedFolderSegmentationDataset(args.data_root, split=args.split, image_size=args.image_size)
    if args.dataset == "mosmed":
        return MosMedTextCSVDataset(args.data_root, split=args.split, image_size=args.image_size)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def score_sample(sample: dict[str, Any], dwt: HaarDWT2D) -> tuple[float, dict[str, Any]]:
    x = sample["image"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)
    with torch.no_grad():
        ll_t, lh_t, hl_t, hh_t = dwt(x)
        mask_small = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
    ll = ll_t[0, 0].numpy()
    lh = lh_t[0, 0].numpy()
    hl = hl_t[0, 0].numpy()
    hh = hh_t[0, 0].numpy()
    lesion = mask_small[0, 0].numpy() > 0.05
    area = float(sample["mask"].float().mean().item())
    hf = np.abs(lh) + np.abs(hl) + np.abs(hh)
    if lesion.any():
        lesion_hf = float(hf[lesion].mean())
    else:
        lesion_hf = 0.0
    # Prefer visible lesions with strong local high-frequency structure, not empty/near-full masks.
    area_weight = min(area / 0.04, 1.0) * min((1.0 - area) / 0.70, 1.0)
    score = lesion_hf * max(area_weight, 0.05)
    payload = {
        "ll": ll,
        "lh": lh,
        "hl": hl,
        "hh": hh,
        "area": area,
        "energies": band_energies(ll, lh, hl, hh),
        "score": score,
    }
    return score, payload


def main() -> None:
    parser = argparse.ArgumentParser("Create paper-style LL/LH/HL/HH wavelet frequency panels.")
    parser.add_argument("--dataset", choices=["qata", "brain", "breast", "prompted", "mosmed"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--indices", type=str, default="", help="Optional comma-separated dataset indices.")
    parser.add_argument("--out-dir", default="generated_figures/frequency_bands")
    args = parser.parse_args()

    ds = load_dataset(args)
    dwt = HaarDWT2D(backend="manual")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.indices.strip():
        chosen = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        scored = []
        for idx in range(len(ds)):
            sample = ds[idx]
            score, payload = score_sample(sample, dwt)
            scored.append((score, idx, payload))
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = [idx for _, idx, _ in scored[: args.num_cases]]

    manifest = []
    for rank, idx in enumerate(chosen, start=1):
        sample = ds[idx]
        _, payload = score_sample(sample, dwt)
        out_path = out_dir / f"{args.dataset}_{args.split}_{rank:02d}_idx{idx:04d}_frequency_bands.png"
        make_panel(sample, payload["ll"], payload["lh"], payload["hl"], payload["hh"], payload["energies"], out_path)
        manifest.append(
            {
                "rank": rank,
                "index": idx,
                "mask_name": str(sample["mask_name"]),
                "lesion_area": payload["area"],
                "score": payload["score"],
                "energies": payload["energies"],
                "path": str(out_path),
            }
        )

    manifest_path = out_dir / f"{args.dataset}_{args.split}_frequency_bands_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} frequency panels to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
