from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class MosMedTextCSVDataset(Dataset):
    """MosMed dataset with per-image text captions stored in split CSV files."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 224,
        max_samples: int | None = None,
        augment: bool = False,
        ct_window: bool = False,
        elastic_prob: float = 0.0,
        elastic_alpha: float = 8.0,
        elastic_sigma: float = 4.0,
        prompt_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.root = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        # CT-style preprocessing & extra aug. Only effective when augment=True
        # (i.e. on train split); ct_window applies to all splits when enabled
        # so train/val/test stay consistent.
        self.ct_window = bool(ct_window)
        self.elastic_prob = float(elastic_prob) if augment else 0.0
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)
        self.prompt_dropout_prob = float(prompt_dropout_prob) if augment else 0.0

        csv_name_map = {
            "train": "Train_text_MosMedData+ 1(in).csv",
            "val": "Val_text_MosMedData+ 1(in).csv",
            "test": "Test_text_MosMedData+(in).csv",
        }
        self.frames_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.csv_path = self.root / csv_name_map[split]

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.frames_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        records: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            if "Image" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(f"Expected columns Image;text in {self.csv_path}")
            for row in reader:
                image_name = row["Image"].strip()
                text = row["text"].strip()
                if not image_name:
                    continue
                image_path = self.frames_dir / image_name
                mask_path = self.masks_dir / image_name
                if not image_path.exists() or not mask_path.exists():
                    continue
                records.append(
                    {
                        "image_name": image_name,
                        "text": text,
                    }
                )

        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise RuntimeError(f"No valid samples found in {self.csv_path}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
        return k / k.sum()

    def _apply_elastic(self, image_arr: np.ndarray, mask_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Torch-only elastic deform: random per-pixel displacement field, blurred
        # by a separable gaussian, then resampled with bilinear (image) / nearest (mask).
        h, w = image_arr.shape
        radius = max(1, int(round(3.0 * self.elastic_sigma)))
        kernel = self._gaussian_kernel1d(self.elastic_sigma, radius)  # (k,)
        k_size = kernel.numel()

        # Random raw displacement field, shape (1, 2, h, w).
        field = torch.empty(1, 2, h, w).uniform_(-1.0, 1.0)
        # Separable conv: horizontal then vertical, per channel.
        kh = kernel.view(1, 1, 1, k_size).expand(2, 1, 1, k_size).contiguous()
        kv = kernel.view(1, 1, k_size, 1).expand(2, 1, k_size, 1).contiguous()
        pad_h = k_size // 2
        field = F.conv2d(F.pad(field, (pad_h, pad_h, 0, 0), mode="reflect"), kh, groups=2)
        field = F.conv2d(F.pad(field, (0, 0, pad_h, pad_h), mode="reflect"), kv, groups=2)
        field = field * self.elastic_alpha  # (1, 2, h, w) — dx, dy in pixels

        # Build sampling grid in [-1, 1] for grid_sample (N=1, H, W, 2).
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        # grid_sample expects (x, y) order; convert pixel coords to [-1, 1].
        dx = field[0, 0]
        dy = field[0, 1]
        gx = 2.0 * (xs + dx) / max(w - 1, 1) - 1.0
        gy = 2.0 * (ys + dy) / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, h, w, 2)

        img_t = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0)
        msk_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
        img_out = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        msk_out = F.grid_sample(msk_t, grid, mode="nearest", padding_mode="reflection", align_corners=True)
        return img_out.squeeze().numpy().astype(np.float32), msk_out.squeeze().numpy().astype(np.float32)

    @staticmethod
    def _ct_histogram_clip(arr: np.ndarray) -> np.ndarray:
        # Per-image robust min/max → pseudo lung-window. Lifts contrast on the
        # thin tissue/lesion band that mid-grey CT slices otherwise compress.
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi - lo < 1e-3:
            return arr
        out = (arr - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _dropout_prompt(text: str) -> str:
        # Prompts are comma-separated clauses. Drop ONE random clause if the
        # text has at least 3 clauses (otherwise it's already minimal).
        parts = [p.strip() for p in re.split(r"[,.]", text) if p.strip()]
        if len(parts) < 3:
            return text
        drop_idx = random.randrange(len(parts))
        kept = [p for i, p in enumerate(parts) if i != drop_idx]
        return ", ".join(kept) + "."

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[index]
        image_name = rec["image_name"]
        image_path = self.frames_dir / image_name
        mask_path = self.masks_dir / image_name

        image = Image.open(image_path).convert("L").resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), resample=Image.NEAREST)

        if self.augment:
            import torchvision.transforms.functional as TF

            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Affine — wider range than before
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                translate = [random.uniform(-0.05, 0.05) * self.image_size, random.uniform(-0.05, 0.05) * self.image_size]
                scale = random.uniform(0.85, 1.15)
                image = TF.affine(image, angle, translate, scale, 0.0, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.affine(mask, angle, translate, scale, 0.0, interpolation=TF.InterpolationMode.NEAREST)

            # Intensity jitter (image only, not mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment and self.elastic_prob > 0 and random.random() < self.elastic_prob:
            image_arr, mask_arr = self._apply_elastic(image_arr, mask_arr)
            mask_arr = (mask_arr > 0.5).astype(np.float32)

        if self.ct_window:
            image_arr = self._ct_histogram_clip(image_arr)

        if self.augment and random.random() > 0.5:
            image_arr = np.clip(
                image_arr + np.random.normal(0, 0.02, image_arr.shape).astype(np.float32),
                0.0, 1.0,
            )

        text_out = rec["text"]
        if self.prompt_dropout_prob > 0 and random.random() < self.prompt_dropout_prob:
            text_out = self._dropout_prompt(text_out)

        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "text": text_out,
            "mask_name": image_name,
        }
