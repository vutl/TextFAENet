from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class QaTaSample:
    image_path: Path
    mask_path: Path
    mask_name: str
    description: str


class QaTaCOV19Dataset(Dataset):
    """QaTa-COV19-v2 dataset reader with segmentation mask and text description."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 224,
        use_text: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.root = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.use_text = use_text
        self.max_samples = max_samples

        split_dir = self.root / ("Train" if split == "train" else "Test")
        self.images_dir = split_dir / "Images"
        self.masks_dir = split_dir / "GTs"
        self.prompt_csv = self.root / "prompt" / f"{split}.csv"

        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Dataset folders/files missing for split={split}. "
                f"Expected: {self.images_dir}, {self.masks_dir}"
            )
        if self.use_text and not self.prompt_csv.exists():
            raise FileNotFoundError(
                f"Prompt CSV missing for split={split}. Expected: {self.prompt_csv}"
            )

        self.samples = self._build_samples()
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def _build_samples(self) -> list[QaTaSample]:
        if not self.use_text:
            samples: list[QaTaSample] = []
            for mask_path in sorted(self.masks_dir.glob("mask_*")):
                if not mask_path.is_file():
                    continue
                image_name = mask_path.name[len("mask_") :]
                image_path = self.images_dir / image_name
                if not image_path.exists():
                    continue
                samples.append(
                    QaTaSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        mask_name=mask_path.name,
                        description="",
                    )
                )

            if not samples:
                raise RuntimeError(f"No valid samples found for split={self.split} in {self.root}")
            return samples

        descriptions: dict[str, str] = {}
        with self.prompt_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Image" not in reader.fieldnames or "Description" not in reader.fieldnames:
                raise ValueError(f"CSV must contain Image and Description columns: {self.prompt_csv}")
            for row in reader:
                descriptions[row["Image"]] = row["Description"]

        samples: list[QaTaSample] = []
        for mask_name, desc in descriptions.items():
            if not mask_name.startswith("mask_"):
                continue

            image_name = mask_name[len("mask_") :]
            image_path = self.images_dir / image_name
            mask_path = self.masks_dir / mask_name
            if not image_path.exists() or not mask_path.exists():
                continue

            samples.append(
                QaTaSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    mask_name=mask_name,
                    description=desc,
                )
            )

        if not samples:
            raise RuntimeError(f"No valid samples found for split={self.split} in {self.root}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).unsqueeze(0)
        return ten

    def _read_mask(self, path: Path) -> torch.Tensor:
        m = Image.open(path).convert("L")
        m = m.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        arr = np.asarray(m, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)
        ten = torch.from_numpy(arr).unsqueeze(0)
        return ten

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[index]
        image = self._read_image(s.image_path)
        mask = self._read_mask(s.mask_path)
        return {
            "image": image,
            "mask": mask,
            "text": s.description,
            "mask_name": s.mask_name,
        }
