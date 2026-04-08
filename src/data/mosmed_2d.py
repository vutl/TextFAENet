from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MosMed2DSegmentationDataset(Dataset):
    """2D slice segmentation dataset prepared from MosMedData 3D NIfTI studies."""

    def __init__(
        self,
        prepared_root: str,
        split: str,
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.root = Path(prepared_root)
        self.image_size = image_size
        self.max_samples = max_samples

        csv_path = self.root / "splits" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        records: list[dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        if self.max_samples is not None:
            records = records[: self.max_samples]

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _to_tensor_gray(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _to_tensor_mask(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[index]
        image_path = self.root / rec["image_path"]
        mask_path = self.root / rec["mask_path"]

        image = Image.open(image_path).convert("L").resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), resample=Image.NEAREST)

        return {
            "image": self._to_tensor_gray(image),
            "mask": self._to_tensor_mask(mask),
            "text": rec.get("prompt", "COVID-19 lesion segmentation"),
            "mask_name": Path(mask_path).name,
            "study_id": rec.get("study_id", ""),
            "ct_class": rec.get("ct_class", ""),
            "slice_idx": rec.get("slice_idx", ""),
        }
