from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PromptedFolderSegmentationDataset(Dataset):
    """Segmentation dataset with split folders and per-split CSV prompts."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.root = Path(root_dir)
        self.image_size = image_size
        self.images_dir = self.root / f"{split}_images"
        self.masks_dir = self.root / f"{split}_masks"
        self.csv_path = self.root / f"{split}.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        records: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Image" not in reader.fieldnames or "Description" not in reader.fieldnames:
                raise ValueError(f"Expected columns Image,Description in {self.csv_path}")
            for row in reader:
                image_name = row["Image"].strip()
                text = row["Description"].strip()
                if not image_name:
                    continue
                image_path = self.images_dir / image_name
                mask_path = self.masks_dir / image_name
                if not image_path.exists() or not mask_path.exists():
                    continue
                records.append({"image_name": image_name, "text": text})

        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise RuntimeError(f"No valid samples found in {self.csv_path}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[index]
        image_name = rec["image_name"]
        image_path = self.images_dir / image_name
        mask_path = self.masks_dir / image_name

        image = Image.open(image_path).convert("L").resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "text": rec["text"],
            "mask_name": image_name,
        }
