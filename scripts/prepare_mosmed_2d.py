from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


def normalize_ct_slice(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32)
    p_lo = np.percentile(x, lo)
    p_hi = np.percentile(x, hi)
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    x = np.clip(x, p_lo, p_hi)
    x = (x - p_lo) / (p_hi - p_lo)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser("Prepare MosMedData 2D slice segmentation dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"datasets/MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0",
    )
    parser.add_argument("--out-root", type=str, default="datasets/mosmed_2d_prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--neg-per-pos",
        type=float,
        default=0.25,
        help="Number of negative slices sampled per positive slice within each study.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.data_root)
    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    split_dir = out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted((root / "masks").glob("study_*_mask.nii"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {(root / 'masks')}.")

    study_records: dict[str, list[dict[str, str]]] = {}

    for mask_path in mask_files:
        study_id = mask_path.name.replace("_mask.nii", "")

        study_candidates = list((root / "studies").glob(f"CT-*/{study_id}.nii"))
        if not study_candidates:
            continue

        study_path = study_candidates[0]
        ct_class = study_path.parent.name

        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(study_path)))  # [Z, H, W]
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))   # [Z, H, W]

        z = min(vol.shape[0], msk.shape[0])
        vol = vol[:z]
        msk = msk[:z]

        pos_indices = [i for i in range(z) if np.any(msk[i] > 0)]
        neg_indices = [i for i in range(z) if not np.any(msk[i] > 0)]

        neg_keep = int(round(len(pos_indices) * args.neg_per_pos))
        if neg_keep > 0 and neg_indices:
            random.shuffle(neg_indices)
            use_indices = sorted(pos_indices + neg_indices[:neg_keep])
        else:
            use_indices = sorted(pos_indices)

        rows: list[dict[str, str]] = []
        for idx in use_indices:
            img_arr = normalize_ct_slice(vol[idx])
            mask_arr = (msk[idx] > 0).astype(np.uint8) * 255

            img_rel = Path("images") / f"{study_id}_z{idx:03d}.png"
            msk_rel = Path("masks") / f"{study_id}_z{idx:03d}_mask.png"

            save_png(out_root / img_rel, img_arr)
            save_png(out_root / msk_rel, mask_arr)

            rows.append(
                {
                    "image_path": img_rel.as_posix(),
                    "mask_path": msk_rel.as_posix(),
                    "study_id": study_id,
                    "slice_idx": str(idx),
                    "ct_class": ct_class,
                    "prompt": "COVID-19 chest CT lesion segmentation",
                    "has_lesion": "1" if idx in pos_indices else "0",
                }
            )

        study_records[study_id] = rows

    study_ids = sorted(study_records.keys())
    random.shuffle(study_ids)

    n = len(study_ids)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_ids = set(study_ids[:n_train])
    val_ids = set(study_ids[n_train : n_train + n_val])
    test_ids = set(study_ids[n_train + n_val :])

    split_map = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    stats = {}
    fieldnames = ["image_path", "mask_path", "study_id", "slice_idx", "ct_class", "prompt", "has_lesion"]
    for split_name, ids in split_map.items():
        rows = [row for sid in ids for row in study_records[sid]]
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        stats[split_name] = {
            "num_studies": len(ids),
            "num_slices": len(rows),
            "num_positive_slices": sum(int(r["has_lesion"]) for r in rows),
        }

    summary = {
        "data_root": str(root),
        "out_root": str(out_root),
        "num_masked_studies": len(study_ids),
        "splits": stats,
        "note": "All 50 segmentation masks in MosMedData belong to a small annotated subset.",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared MosMed 2D dataset at: {out_root}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
