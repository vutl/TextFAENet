from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def read_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate binary predicted-mask folder against GT folder.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--pred-prefix-strip", type=str, default="")
    parser.add_argument("--gt-prefix-add", type=str, default="")
    args = parser.parse_args()

    pred_paths = sorted([p for p in args.pred_dir.iterdir() if p.is_file()])
    rows = []
    total_inter = total_union = total_pred = total_target = 0.0
    eps = 1e-6
    for pred_path in pred_paths:
        name = pred_path.name
        gt_name = name
        if args.pred_prefix_strip and gt_name.startswith(args.pred_prefix_strip):
            gt_name = gt_name[len(args.pred_prefix_strip):]
        gt_name = args.gt_prefix_add + gt_name
        gt_path = args.gt_dir / gt_name
        if not gt_path.exists():
            continue
        gt_img = Image.open(gt_path).convert("L")
        gt = (np.asarray(gt_img, dtype=np.float32) > 127).astype(np.float32)
        pred = read_mask(pred_path, size=gt_img.size)
        inter = float((pred * gt).sum())
        pred_sum = float(pred.sum())
        target_sum = float(gt.sum())
        union = float(((pred + gt) > 0).sum())
        dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
        iou = (inter + eps) / (union + eps)
        rows.append(
            {
                "mask_name": gt_name,
                "dice": dice,
                "iou": iou,
                "intersection": inter,
                "union": union,
                "pred_pixels": pred_sum,
                "target_pixels": target_sum,
            }
        )
        total_inter += inter
        total_union += union
        total_pred += pred_sum
        total_target += target_sum

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mask_name", "dice", "iou", "intersection", "union", "pred_pixels", "target_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "num_images": len(rows),
        "per_image_dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "per_image_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
        "global_dice": float((2.0 * total_inter + eps) / (total_pred + total_target + eps)),
        "global_iou": float((total_inter + eps) / (total_union + eps)),
        "per_image_csv": str(args.output_csv),
    }
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
