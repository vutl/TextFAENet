from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset


def to_rgb_gray(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    g = (x * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    gray = (mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def compute_mask_areas(dataset: QaTaCOV19Dataset) -> list[float]:
    areas: list[float] = []
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"].squeeze(0).numpy()
        areas.append(float(mask.mean()))
    return areas


def select_diverse_indices(areas: list[float], num_samples: int) -> list[int]:
    positive = [(idx, area) for idx, area in enumerate(areas) if area > 0.0]
    if not positive:
        return list(range(min(num_samples, len(areas))))

    positive.sort(key=lambda x: x[1])
    if num_samples == 1:
        return [positive[len(positive) // 2][0]]

    chosen: list[int] = []
    for pos in range(num_samples):
        frac = pos / max(num_samples - 1, 1)
        pick = positive[int(round(frac * (len(positive) - 1)))][0]
        if pick not in chosen:
            chosen.append(pick)

    cursor = 0
    while len(chosen) < min(num_samples, len(areas)):
        if cursor not in chosen:
            chosen.append(cursor)
        cursor += 1
    return chosen


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, fill: tuple[int, int, int]) -> None:
    wrapped = textwrap.fill(text, width=width_chars)
    draw.multiline_text(xy, wrapped, fill=fill, spacing=4)


def main() -> None:
    parser = argparse.ArgumentParser("Create a compact dataset-overview figure for QaTa-COV19-v2")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out-path", type=str, default="paper_figures/fig2_qata_dataset_overview.png")
    args = parser.parse_args()

    ds = QaTaCOV19Dataset(root_dir=args.data_root, split=args.split, image_size=args.image_size, use_text=True)
    areas = compute_mask_areas(ds)
    indices = select_diverse_indices(areas, args.num_samples)

    cell_w = args.image_size
    cell_h = args.image_size
    left_w = 360
    cols = 3
    margin = 24
    row_gap = 18
    title_h = 72
    header_h = 28
    footer_h = 42
    row_h = header_h + cell_h

    canvas_w = margin * 2 + left_w + cols * cell_w
    canvas_h = title_h + margin + len(indices) * row_h + max(len(indices) - 1, 0) * row_gap + footer_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    out = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(out)

    draw.text((margin, 16), "Figure 2. Dataset Overview on QaTa-COV19-v2", fill=(15, 15, 15))
    draw.text(
        (margin, 40),
        "Chest X-ray samples with paired infection masks and free-text descriptions.",
        fill=(80, 80, 80),
    )

    col_x = [margin + left_w + i * cell_w for i in range(cols)]
    for x, title in zip(col_x, ["Input", "GT Overlay", "GT Mask"]):
        draw.text((x + 8, title_h), title, fill=(20, 20, 20))

    y = title_h + margin
    for row_id, idx in enumerate(indices):
        sample = ds[idx]
        img = sample["image"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()
        prompt = sample["text"].strip()
        area_pct = mask.mean() * 100.0

        base = to_rgb_gray(img)
        overlay = overlay_mask(base, mask > 0.5, color=(220, 40, 40), alpha=0.45)
        mask_rgb = mask_to_rgb(mask)

        draw.text((margin, y), f"Sample {row_id + 1}", fill=(15, 15, 15))
        draw.text((margin, y + 20), f"Split: {args.split}", fill=(80, 80, 80))
        draw.text((margin + 110, y + 20), f"Mask area: {area_pct:.1f}%", fill=(80, 80, 80))
        draw_wrapped_text(draw, (margin, y + 44), f"Prompt: {prompt}", width_chars=52, fill=(35, 35, 35))

        row_y = y + header_h
        out.paste(Image.fromarray(base), (col_x[0], row_y))
        out.paste(Image.fromarray(overlay), (col_x[1], row_y))
        out.paste(Image.fromarray(mask_rgb), (col_x[2], row_y))

        y += row_h + row_gap

    footer_y = canvas_h - footer_h + 8
    draw.text(
        (margin, footer_y),
        "Modality: Chest X-ray | Target: COVID-19 infection region | Supervision: image-mask-text pairs",
        fill=(70, 70, 70),
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Saved dataset overview figure to: {out_path}")


if __name__ == "__main__":
    main()
