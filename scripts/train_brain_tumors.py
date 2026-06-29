from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve()
TEXTFAENET_ROOT = ROOT.parents[1]
if str(TEXTFAENET_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXTFAENET_ROOT))

from src.models import LFAENetTGFSv3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.7,
        pos_weight: float | None = None,
        boundary_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pred_dilated = F.max_pool2d(probs, kernel_size=3, stride=1, padding=1)
        pred_eroded = -F.max_pool2d(-probs, kernel_size=3, stride=1, padding=1)
        pred_boundary = pred_dilated - pred_eroded

        target_dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
        target_eroded = -F.max_pool2d(-targets, kernel_size=3, stride=1, padding=1)
        target_boundary = target_dilated - target_eroded

        return F.mse_loss(pred_boundary, target_boundary)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        loss = self.bce_weight * bce + self.dice_weight * self.dice_loss(logits, targets)
        if self.boundary_weight > 0:
            loss = loss + self.boundary_weight * self.boundary_loss(logits, targets)
        return loss


def _parse_class_from_description(text: str) -> int:
    """Return class id parsed from CSV Description prefix.

    0 = Unilateral, 1 = Bilateral, -1 = unknown (counted as its own bucket).
    """
    head = text.strip().lower()
    if head.startswith("bilateral"):
        return 1
    if head.startswith("unilateral"):
        return 0
    return -1


def _swap_lr_in_text(text: str) -> str:
    """Swap occurrences of 'left' and 'right' so flipped image still matches text."""
    def swap(match) -> str:
        word = match.group(0)
        replacement = "right" if word.lower() == "left" else "left"
        return replacement.capitalize() if word[0].isupper() else replacement

    import re

    return re.sub(r"\b(left|right)\b", swap, text, flags=re.IGNORECASE)


class CsvPromptedFolderSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 224,
        max_samples: int | None = None,
        csv_path: str | None = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.root = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.images_dir = self.root / f"{split}_images"
        self.masks_dir = self.root / f"{split}_masks"
        if csv_path is None:
            csv_path = str(self.root / f"{split}.csv")
        self.csv_path = Path(csv_path)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        records: list[dict] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Image" not in reader.fieldnames or "Description" not in reader.fieldnames:
                raise ValueError(f"Expected columns Image,Description in {self.csv_path}")
            for row in reader:
                image_name = str(row["Image"]).strip()
                text = str(row["Description"]).strip()
                if not image_name:
                    continue
                image_path = self.images_dir / image_name
                mask_path = self.masks_dir / image_name
                if not image_path.exists() or not mask_path.exists():
                    continue
                cls_id = _parse_class_from_description(text)
                records.append({"image_name": image_name, "text": text, "class": cls_id})

        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise RuntimeError(f"No valid samples found in {self.csv_path}")

        self.records = records

    def get_class_labels(self) -> list[int]:
        return [r["class"] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[index]
        image_name = rec["image_name"]
        image_path = self.images_dir / image_name
        mask_path = self.masks_dir / image_name
        text = rec["text"]

        from PIL import Image

        image = Image.open(image_path).convert("L").resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), resample=Image.NEAREST)

        if self.augment:
            from PIL import ImageEnhance
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                text = _swap_lr_in_text(text)
            if random.random() < 0.6:
                angle = random.uniform(-15.0, 15.0)
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                image = ImageEnhance.Brightness(image).enhance(factor)
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                image = ImageEnhance.Contrast(image).enhance(factor)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                gamma = random.uniform(0.8, 1.25)
                image_arr = np.clip(image_arr ** gamma, 0.0, 1.0).astype(np.float32)
            if random.random() < 0.4:
                noise = np.random.normal(loc=0.0, scale=0.02, size=image_arr.shape).astype(np.float32)
                image_arr = np.clip(image_arr + noise, 0.0, 1.0)

        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "text": text,
            "mask_name": image_name,
        }


class TextSegCollator:
    def __init__(
        self,
        tokenizer=None,
        max_length: int = 64,
        prompt_source: str = "xlsx",
        fixed_prompt: str = "Segment the tumor region.",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_source = prompt_source
        self.fixed_prompt = fixed_prompt

    def __call__(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        masks = torch.stack([x["mask"] for x in batch], dim=0)
        texts = [str(x.get("text", "")) for x in batch]
        if self.prompt_source == "fixed":
            texts = [self.fixed_prompt for _ in texts]
        names = [x["mask_name"] for x in batch]

        out = {
            "image": images,
            "mask": masks,
            "text": texts,
            "mask_name": names,
        }

        if self.tokenizer is not None:
            toks = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out["input_ids"] = toks["input_ids"]
            out["attention_mask"] = toks["attention_mask"]

        return out


def create_model(args, device: torch.device):
    tokenizer = None
    text_encoder_type = "simple"
    if args.use_cxr_bert:
        text_encoder_type = "biomedvlp-cxr-bert"

    model = LFAENetTGFSv3(
            in_channels=1,
            num_classes=1,
            text_dim=args.text_dim,
            vocab_size=args.vocab_size,
            text_encoder_type=text_encoder_type,
            text_backbone_path=args.cxr_bert_dir,
            freeze_text_backbone=args.freeze_text_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            fusion_mode=args.fusion_mode,
            drop_hh_in_decoder=args.drop_hh_in_decoder,
            hh_drop_mode=args.hh_drop_mode,
            low_level_hf_scale=args.low_level_hf_scale,
            learnable_low_level_hf_scale=args.learnable_low_level_hf_scale,
            spatial_sharpen_power=args.spatial_sharpen_power,
            learnable_spatial_sharpen=args.learnable_spatial_sharpen,
            use_deep_supervision=args.use_deep_supervision,
            encoder_text_fusion=args.encoder_text_fusion,
            norm_type=args.norm_type,
            conv_block_depth=args.conv_block_depth,
            dropout_p=args.dropout_p,
            grounding_n_heads=args.grounding_n_heads,
            encoder_type=getattr(args, "encoder_type", "from_scratch"),
            pretrained_image_encoder=getattr(args, "pretrained_image_encoder", True),
            freeze_encoder_bn=getattr(args, "freeze_encoder_bn", True),
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.cxr_bert_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    return model.to(device), tokenizer


def compute_foreground_stats(
    root_dir: str,
    split: str,
    max_samples: int | None = None,
    csv_path: str | None = None,
) -> dict[str, float]:
    root = Path(root_dir)
    if csv_path is None:
        csv_path = str(root / f"{split}.csv")
    csv_path = Path(csv_path)

    rows: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "Image" not in reader.fieldnames:
            raise ValueError(f"Expected column Image in {csv_path}")
        for row in reader:
            rows.append(str(row["Image"]).strip())
    if max_samples is not None:
        rows = rows[:max_samples]

    fg_pixels = 0.0
    total_pixels = 0.0
    non_empty = 0
    areas: list[float] = []
    masks_dir = root / f"{split}_masks"
    from PIL import Image

    for name in rows:
        if not name:
            continue
        mask_path = masks_dir / name
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) > 127
        area = float(mask.mean())
        fg_pixels += float(mask.sum())
        total_pixels += float(mask.size)
        non_empty += int(mask.any())
        areas.append(area)

    fg_fraction = fg_pixels / max(total_pixels, 1.0)
    return {
        "num_samples": float(len(rows)),
        "fg_fraction": float(fg_fraction),
        "non_empty_fraction": float(non_empty / max(len(rows), 1)),
        "avg_area": float(np.mean(areas)) if areas else 0.0,
        "median_area": float(np.median(areas)) if areas else 0.0,
    }


def auto_pos_weight_from_stats(stats: dict[str, float], max_pos_weight: float) -> float:
    fg = max(stats["fg_fraction"], 1e-6)
    neg = max(1.0 - fg, 1e-6)
    return min(neg / fg, max_pos_weight)


def parse_thresholds(spec: str) -> list[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    values = [x for x in values if 0.0 < x < 1.0]
    if not values:
        raise ValueError("No valid thresholds parsed; expected comma-separated values in (0,1).")
    return values


def compute_loss_with_aux(
    criterion,
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: dict | None,
    aux_w_d4: float,
    aux_w_d3: float,
    aux_w_d2: float,
    grounding_loss_weight: float = 0.0,
) -> torch.Tensor:
    loss = criterion(logits, targets)
    if aux is None:
        return loss
    if "d4" in aux:
        loss = loss + aux_w_d4 * criterion(aux["d4"], targets)
    if "d3" in aux:
        loss = loss + aux_w_d3 * criterion(aux["d3"], targets)
    if "d2" in aux:
        loss = loss + aux_w_d2 * criterion(aux["d2"], targets)
    if grounding_loss_weight > 0 and "grounding" in aux and len(aux["grounding"]) > 0:
        gmap = aux["grounding"]
        grounding = 0.0
        for _, mask_raw in gmap.items():
            mask_up = F.interpolate(mask_raw, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            mask_up = mask_up.float().clamp(1e-7, 1.0 - 1e-7)
            target_fp32 = targets.float()
            grounding_bce = -(
                target_fp32 * torch.log(mask_up)
                + (1.0 - target_fp32) * torch.log1p(-mask_up)
            ).mean()
            grounding = grounding + grounding_bce
        loss = loss + grounding_loss_weight * grounding / len(gmap)
    return loss


def forward_model(batch, model, args, device: torch.device):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    need_aux = args.use_deep_supervision or getattr(args, "grounding_loss_weight", 0.0) > 0
    if need_aux:
        logits, aux = model(image, token_ids=input_ids, attention_mask=attention_mask, return_aux=True)
    else:
        logits = model(image, token_ids=input_ids, attention_mask=attention_mask)
        aux = None

    return mask, logits, aux


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None, threshold: float = 0.5):
    from tqdm import tqdm

    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    metric_totals = new_metric_totals()

    accum_steps = max(1, int(args.grad_accum_steps))
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    phase = "train" if train_mode else "val"
    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"  [{phase}]", leave=False, dynamic_ncols=True)

    for step, batch in pbar:
        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                mask, logits, aux = forward_model(batch, model, args, device)
                loss = compute_loss_with_aux(
                    criterion,
                    logits,
                    mask,
                    aux,
                    args.aux_w_d4,
                    args.aux_w_d3,
                    args.aux_w_d2,
                    grounding_loss_weight=getattr(args, "grounding_loss_weight", 0.0),
                )

            if train_mode:
                scaled_loss = loss / accum_steps
                max_grad_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
                if args.use_amp:
                    scaler.scale(scaled_loss).backward()
                    if step % accum_steps == 0 or step == len(loader):
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                (p for g in optimizer.param_groups for p in g["params"]),
                                max_norm=max_grad_norm,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if step % accum_steps == 0 or step == len(loader):
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                (p for g in optimizer.param_groups for p in g["params"]),
                                max_norm=max_grad_norm,
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        m = batch_metrics(logits.detach(), mask, threshold=threshold)
        total_loss += loss.item() * m["sample_count"]
        update_metric_totals(metric_totals, m)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.4f}", refresh=False)

    return finalize_metric_totals(metric_totals, total_loss)


@torch.no_grad()
def run_test_with_tta(model, loader, criterion, device, args, tokenizer, threshold: float):
    """Evaluate test set with horizontal-flip TTA. Text is also l↔r swapped to
    stay consistent with the flipped image. Original + flipped probabilities are
    averaged, then re-thresholded.
    """
    model.eval()
    total_loss = 0.0
    metric_totals = new_metric_totals()

    for batch in loader:
        # 1) Original forward
        mask, logits_orig, _ = forward_model(batch, model, args, device)
        probs_orig = torch.sigmoid(logits_orig)

        # 2) Flip image + swap l/r in text + re-tokenize
        flipped_image = torch.flip(batch["image"], dims=[-1])
        flipped_batch = {
            "image": flipped_image,
            "mask": batch["mask"],
            "mask_name": batch.get("mask_name", []),
        }
        if tokenizer is not None and "input_ids" in batch:
            flipped_texts = [_swap_lr_in_text(str(t)) for t in batch.get("text", [])]
            toks = tokenizer(
                flipped_texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_text_len,
                return_tensors="pt",
            )
            flipped_batch["input_ids"] = toks["input_ids"]
            flipped_batch["attention_mask"] = toks["attention_mask"]
            flipped_batch["text"] = flipped_texts
        else:
            flipped_batch["text"] = batch.get("text", [])
            if "input_ids" in batch:
                flipped_batch["input_ids"] = batch["input_ids"]
                flipped_batch["attention_mask"] = batch["attention_mask"]

        _, logits_flip, _ = forward_model(flipped_batch, model, args, device)
        probs_flip = torch.sigmoid(torch.flip(logits_flip, dims=[-1]))

        # 3) Average probabilities then re-derive logits
        avg_probs = 0.5 * (probs_orig + probs_flip)
        avg_probs = avg_probs.clamp(1e-7, 1.0 - 1e-7)
        avg_logits = torch.log(avg_probs / (1.0 - avg_probs))

        loss = criterion(avg_logits, mask).item()
        m = batch_metrics(avg_logits, mask, threshold=threshold)
        total_loss += loss * m["sample_count"]
        update_metric_totals(metric_totals, m)

    return finalize_metric_totals(metric_totals, total_loss)


@torch.no_grad()
def evaluate_thresholds(model, loader, criterion, device, args, thresholds: list[float]) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    metric_totals = {thr: new_metric_totals() for thr in thresholds}
    loss_sums = {thr: 0.0 for thr in thresholds}

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            mask, logits, aux = forward_model(batch, model, args, device)
            loss = compute_loss_with_aux(
                criterion,
                logits,
                mask,
                aux,
                args.aux_w_d4,
                args.aux_w_d3,
                args.aux_w_d2,
                grounding_loss_weight=getattr(args, "grounding_loss_weight", 0.0),
            ).item()

        for thr in thresholds:
            m = batch_metrics(logits, mask, threshold=thr)
            loss_sums[thr] += loss * m["sample_count"]
            update_metric_totals(metric_totals[thr], m)

    results = {
        thr: finalize_metric_totals(metric_totals[thr], loss_sums[thr])
        for thr in thresholds
    }

    best_threshold = max(thresholds, key=lambda thr: results[thr]["dice"])
    return results, best_threshold


def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    tgt_sum = targets.sum(dim=(1, 2, 3))

    per_image_iou = (inter + eps) / (union + eps)
    per_image_dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    return {
        "sample_count": float(targets.shape[0]),
        "iou": per_image_iou.mean().item(),
        "dice": per_image_dice.mean().item(),
        "iou_sum": per_image_iou.sum().item(),
        "dice_sum": per_image_dice.sum().item(),
        "intersection_sum": inter.sum().item(),
        "union_sum": union.sum().item(),
        "pred_sum": pred_sum.sum().item(),
        "target_sum": tgt_sum.sum().item(),
        "pixel_count": float(targets.numel()),
    }


def new_metric_totals() -> dict[str, float]:
    return {
        "sample_count": 0.0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
        "intersection_sum": 0.0,
        "union_sum": 0.0,
        "pred_sum": 0.0,
        "target_sum": 0.0,
        "pixel_count": 0.0,
    }


def update_metric_totals(totals: dict[str, float], batch: dict[str, float]) -> None:
    for key in totals:
        totals[key] += float(batch[key])


def finalize_metric_totals(
    totals: dict[str, float],
    loss_sum: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    sample_count = max(totals["sample_count"], 1.0)
    pixel_count = max(totals["pixel_count"], 1.0)
    per_image_iou = totals["iou_sum"] / sample_count
    per_image_dice = totals["dice_sum"] / sample_count
    return {
        "loss": loss_sum / sample_count,
        "iou": per_image_iou,
        "dice": per_image_dice,
        "per_image_iou": per_image_iou,
        "per_image_dice": per_image_dice,
        "global_iou": (totals["intersection_sum"] + eps) / (totals["union_sum"] + eps),
        "global_dice": (2.0 * totals["intersection_sum"] + eps)
        / (totals["pred_sum"] + totals["target_sum"] + eps),
        "pred_pos_ratio": totals["pred_sum"] / pixel_count,
        "gt_pos_ratio": totals["target_sum"] / pixel_count,
        "num_images": int(totals["sample_count"]),
    }


def poly_lr(base_lr: float, epoch: int, max_epochs: int, power: float) -> float:
    return base_lr * ((1.0 - (epoch / max_epochs)) ** power)


def cosine_lr(base_lr: float, epoch: int, max_epochs: int, min_lr: float) -> float:
    if max_epochs <= 1:
        return base_lr
    t = epoch / (max_epochs - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t))


def lr_with_warmup(
    base_lr: float,
    epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    scheduler: str,
    min_lr: float,
    poly_power: float,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Linear warmup from min_lr → base_lr across warmup_epochs.
        frac = (epoch + 1) / max(warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * frac
    adj_epoch = epoch - warmup_epochs
    adj_total = max(max_epochs - warmup_epochs, 1)
    if scheduler == "cosine":
        return cosine_lr(base_lr, adj_epoch, adj_total, min_lr)
    return poly_lr(base_lr, adj_epoch, adj_total, poly_power)


def build_balanced_sampler(dataset) -> WeightedRandomSampler | None:
    labels = dataset.get_class_labels()
    if not labels:
        return None
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        return None
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def append_log_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state_dict(model: nn.Module, args) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if args.use_cxr_bert and args.freeze_text_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("text_encoder.model.")}
    return state


def apply_experiment_preset(args) -> None:
    if args.experiment == "v3_resnet50_cxr":
        # v5 (ResNet-50 + higher resolution) + STABILITY recipe.
        # v5 diverged at epoch 16 because the single global lr (1.5e-4) was too
        # high for the pretrained encoder. v6 fixes this by:
        #   * Differential LR: encoder gets 10x smaller lr than decoder/fusion.
        #   * Lower base lr for the new modules (1e-4 instead of 1.5e-4).
        #   * Longer warmup (10 epochs) for stable startup.
        #   * Gradient clipping (max_grad_norm=1.0) to block gradient explosions.
        #   * AdamW eps=1e-6 for better numerical stability.
        #
        # Backup ideas (NOT applied here; enable if v6 still falls short of 0.86):
        #   Option 2 — reduce model capacity:
        #     * Skip FreqA at the deepest non-bottleneck stage (s3, 1024 ch).
        #       Save ~55M params, reduce overfit risk.
        #     * Or: reduce decoder out channels (dec4=256, dec3=128, dec2=64, dec1=32).
        #   Option 3 — better loss for small/imbalanced targets:
        #     * Replace BCE+Dice with Focal Tversky (α=0.7, β=0.3, γ=0.75).
        #     * Or add it as third term: bce_w=0.1, dice_w=0.5, focal_tversky_w=0.4.
        args.model_type = "lfaenet_tgfs_v3"
        args.prompt_source = "csv"
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.fusion_mode = "both"
        args.encoder_text_fusion = "cross_attn"
        args.hh_drop_mode = "learned"
        args.low_level_hf_scale = 0.6
        args.learnable_low_level_hf_scale = True
        args.spatial_sharpen_power = 2.0
        args.learnable_spatial_sharpen = True
        args.use_deep_supervision = True
        args.augment_train = True
        args.norm_type = "gn"
        args.conv_block_depth = 3
        args.dropout_p = 0.1
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.3
        args.boundary_weight = 0.1
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.max_pos_weight = 16.0
        args.weight_decay = 5e-4
        args.early_stop_patience = 20
        args.balanced_sampling = True
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        args.image_size = 320
        args.batch_size = 2
        args.grad_accum_steps = 4
        # Stability recipe.
        args.lr = 1e-4
        args.encoder_lr = 1e-5
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 10
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser("Train Text-FAENet v3 on brain_tumors")
    parser.add_argument("--data-root", type=str, default=str(TEXTFAENET_ROOT / "datasets" / "brain_tumors"))
    parser.add_argument("--train-csv-path", type=str, default=None)
    parser.add_argument("--val-csv-path", type=str, default=None)
    parser.add_argument("--test-csv-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=str(TEXTFAENET_ROOT / "runs" / "brain_tumors_v3_resnet50_cxr_seed42"))
    parser.add_argument("--experiment", type=str, choices=["v3_resnet50_cxr"], default="v3_resnet50_cxr")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, choices=["poly", "cosine"], default="cosine")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument(
        "--preset-early-stop-patience",
        type=int,
        default=None,
        help="Override the experiment preset early-stop patience after the preset is applied.",
    )

    parser.add_argument("--use-cxr-bert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cxr-bert-dir", type=str, default=str(TEXTFAENET_ROOT / "BiomedVLP-CXR-BERT-specialized"))
    parser.add_argument("--freeze-text-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-hh-in-decoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hh-drop-mode", type=str, choices=["zero", "keep", "learned"], default="zero")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--fusion-mode", type=str, choices=["decoder", "both"], default="decoder")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--use-deep-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-w-d4", type=float, default=0.4)
    parser.add_argument("--aux-w-d3", type=float, default=0.6)
    parser.add_argument("--aux-w-d2", type=float, default=0.8)
    parser.add_argument("--low-level-hf-scale", type=float, default=0.6)
    parser.add_argument("--learnable-low-level-hf-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spatial-sharpen-power", type=float, default=2.0)
    parser.add_argument("--learnable-spatial-sharpen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-text-fusion", type=str, choices=["film", "cross_attn"], default="film")
    parser.add_argument(
        "--dwt-strategy",
        type=str,
        choices=["upsample", "pad_crop", "lowres_conv_bottleneck", "lowres_pad_crop", "stem_dwt"],
        default="lowres_pad_crop",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--save-debug-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-source", type=str, choices=["csv", "fixed"], default="csv")
    parser.add_argument("--fixed-prompt", type=str, default="Segment the tumor region.")
    parser.add_argument("--augment-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-type", type=str, choices=["bn", "gn"], default="bn")
    parser.add_argument("--conv-block-depth", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--grounding-n-heads", type=int, default=1)
    parser.add_argument("--grounding-loss-weight", type=float, default=0.0)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--use-tta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-type", type=str, choices=["from_scratch", "resnet50"], default="from_scratch")
    parser.add_argument("--pretrained-image-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder-bn", action=argparse.BooleanOptionalAction, default=True)
    # Stability knobs for fine-tuning pretrained encoders.
    parser.add_argument("--encoder-lr", type=float, default=0.0,
                        help="If >0 and encoder_type=resnet50, use a separate lr for the image_encoder param group.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="If >0, clip global grad norm to this value before each optimizer step.")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="AdamW eps (set 1e-6 for better numerical stability on MPS / mixed precision).")
    parser.add_argument("--train-on-trainval", action=argparse.BooleanOptionalAction, default=False,
                        help="Combine train+val for training. val_loader uses test split for monitoring. No early stopping.")

    args = parser.parse_args()
    apply_experiment_preset(args)
    if args.preset_early_stop_patience is not None:
        args.early_stop_patience = int(args.preset_early_stop_patience)
    if args.smoke_test:
        args.epochs = 1
        args.image_size = 224
        args.batch_size = 1
        args.grad_accum_steps = 1
        args.max_train_samples = 2
        args.max_val_samples = 2
        args.max_test_samples = 2
        args.augment_train = False
        args.balanced_sampling = False
        args.use_tta = False
        args.early_stop_patience = 0

    if args.drop_hh_in_decoder is not None:
        args.hh_drop_mode = "zero" if args.drop_hh_in_decoder else "keep"
    if args.unfreeze_last_n < 0:
        raise ValueError("--unfreeze-last-n must be >= 0")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    args.save_last_every = max(1, int(args.save_last_every))
    if args.train_on_trainval:
        raise ValueError("Paper protocol requires a separate validation split; --train-on-trainval is disabled.")

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda"

    train_ds = CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="train",
        image_size=args.image_size,
        max_samples=args.max_train_samples,
        csv_path=args.train_csv_path,
        augment=args.augment_train,
    )
    val_ds = CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="val",
        image_size=args.image_size,
        max_samples=args.max_val_samples,
        csv_path=args.val_csv_path,
    )
    test_ds = CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="test",
        image_size=args.image_size,
        max_samples=args.max_test_samples,
        csv_path=args.test_csv_path,
    )

    train_on_trainval = bool(getattr(args, "train_on_trainval", False))
    if train_on_trainval:
        # Augment val too and concatenate with train for a larger training set.
        val_aug_ds = CsvPromptedFolderSegmentationDataset(
            root_dir=args.data_root,
            split="val",
            image_size=args.image_size,
            max_samples=args.max_val_samples,
            csv_path=args.val_csv_path,
            augment=args.augment_train,
        )
        from torch.utils.data import ConcatDataset
        effective_train_ds = ConcatDataset([train_ds, val_aug_ds])
        monitoring_ds = test_ds  # test split used for per-epoch monitoring only
    else:
        effective_train_ds = train_ds
        monitoring_ds = val_ds

    train_mask_stats = compute_foreground_stats(
        args.data_root,
        "train",
        max_samples=args.max_train_samples,
        csv_path=args.train_csv_path,
    )
    if args.pos_weight.lower() == "auto":
        resolved_pos_weight = auto_pos_weight_from_stats(train_mask_stats, args.max_pos_weight)
    else:
        resolved_pos_weight = float(args.pos_weight)
    args.resolved_pos_weight = resolved_pos_weight
    args.train_fg_fraction = train_mask_stats["fg_fraction"]
    args.train_non_empty_fraction = train_mask_stats["non_empty_fraction"]
    args.train_avg_mask_area = train_mask_stats["avg_area"]

    model, tokenizer = create_model(args, device)
    collate_fn = TextSegCollator(
        tokenizer=tokenizer,
        max_length=args.max_text_len,
        prompt_source=args.prompt_source,
        fixed_prompt=args.fixed_prompt,
    )
    if train_on_trainval and getattr(args, "balanced_sampling", False):
        # ConcatDataset doesn't have get_class_labels(); build from components.
        combined_labels = train_ds.get_class_labels() + val_aug_ds.get_class_labels()
        counts: dict[int, int] = {}
        for lbl in combined_labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        if len(counts) >= 2:
            from torch.utils.data import WeightedRandomSampler
            weights = [1.0 / counts[lbl] for lbl in combined_labels]
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        else:
            train_sampler = None
    else:
        train_sampler = build_balanced_sampler(effective_train_ds) if getattr(args, "balanced_sampling", False) else None
    train_loader = DataLoader(
        effective_train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        monitoring_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    criterion = SegLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=resolved_pos_weight,
        boundary_weight=args.boundary_weight,
    )
    # Build param groups. When using a pretrained image encoder (ResNet-50)
    # together with a positive --encoder-lr, give it its own param group with a
    # separate (smaller) learning rate. Each group stores its own _base_lr and
    # _min_lr so the LR scheduler can warm up / decay them independently while
    # keeping their relative ratio constant.
    encoder_lr = float(getattr(args, "encoder_lr", 0.0) or 0.0)
    use_separate_encoder_lr = (
        encoder_lr > 0
        and getattr(args, "encoder_type", "from_scratch") == "resnet50"
        and hasattr(model, "image_encoder")
    )
    if use_separate_encoder_lr:
        encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        enc_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_param_ids]
        # Encoder min_lr scales by the same ratio as base lr.
        ratio = encoder_lr / max(args.lr, 1e-12)
        encoder_min_lr = args.min_lr * ratio
        param_groups = [
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": args.weight_decay,
                "_base_lr": encoder_lr,
                "_min_lr": encoder_min_lr,
                "_group_name": "image_encoder",
            },
            {
                "params": other_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "_base_lr": args.lr,
                "_min_lr": args.min_lr,
                "_group_name": "main",
            },
        ]
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "_base_lr": args.lr,
                "_min_lr": args.min_lr,
                "_group_name": "all",
            },
        ]
    optim_eps = float(getattr(args, "optim_eps", 1e-8) or 1e-8)
    if args.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            momentum=args.momentum,
            nesterov=False,
        )
    else:
        optimizer = AdamW(param_groups, eps=optim_eps)
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    thresholds = parse_thresholds(args.metric_thresholds)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_log_path = save_dir / "epoch_log.txt"
    history_path = save_dir / "history.json"
    final_test_txt_path = save_dir / "final_test.txt"

    start_epoch = 1
    base_phase_lr = args.lr
    best_dice = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []
    no_improve_epochs = 0

    if args.resume_existing and args.resume_ckpt is None:
        last_path = save_dir / "last.pt"
        best_path = save_dir / "best.pt"
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            args.resume_ckpt = str(resume_path)

    if args.resume_ckpt is None:
        txt_log_path.write_text("", encoding="utf-8")
        final_test_txt_path.write_text("", encoding="utf-8")
    else:
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            base_phase_lr = optimizer.param_groups[0]["lr"]
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", best_dice))
        best_threshold = float(ckpt.get("best_threshold", best_threshold))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", no_improve_epochs))
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if "no_improve_epochs" not in ckpt and history:
                last_best_idx = None
                for idx, hist_row in enumerate(history):
                    if abs(float(hist_row.get("val_dice", -1.0)) - best_dice) <= 1e-12:
                        last_best_idx = idx
                if last_best_idx is not None:
                    no_improve_epochs = max(0, len(history) - last_best_idx - 1)

    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    append_log_line(txt_log_path, f"save_dir={save_dir}")
    append_log_line(txt_log_path, f"device={device}")
    append_log_line(
        txt_log_path,
        (
            f"train_samples={len(effective_train_ds)}"
            + (f" (trainval combined: {len(train_ds)}+{len(val_aug_ds)})" if train_on_trainval else "")
            + f" monitoring_samples={len(monitoring_ds)} test_samples={len(test_ds)}"
        ),
    )
    append_log_line(
        txt_log_path,
        (
            "train_mask_stats "
            f"fg_fraction={train_mask_stats['fg_fraction']:.6f} "
            f"non_empty_fraction={train_mask_stats['non_empty_fraction']:.6f} "
            f"avg_area={train_mask_stats['avg_area']:.6f} "
            f"median_area={train_mask_stats['median_area']:.6f} "
            f"resolved_pos_weight={resolved_pos_weight:.4f}"
        ),
    )
    append_log_line(txt_log_path, json.dumps(vars(args), ensure_ascii=False))

    end_epoch = args.epochs
    total_phase_epochs = max(args.epochs, 1)

    warmup_epochs = max(0, int(getattr(args, "lr_warmup_epochs", 0)))
    for epoch in range(start_epoch, end_epoch + 1):
        phase_epoch = epoch - 1
        # Schedule each param group independently using its own _base_lr / _min_lr,
        # so a smaller encoder lr stays proportionally smaller across warmup+decay.
        for pg in optimizer.param_groups:
            base_lr_g = float(pg.get("_base_lr", pg["lr"]))
            min_lr_g = float(pg.get("_min_lr", args.min_lr))
            pg["lr"] = lr_with_warmup(
                base_lr_g,
                phase_epoch,
                total_phase_epochs,
                warmup_epochs,
                args.lr_scheduler,
                min_lr_g,
                args.poly_power,
            )
        # Primary lr used for logging (last group is "main" / "all").
        lr = optimizer.param_groups[-1]["lr"]

        train_stats = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args,
            optimizer=optimizer,
            scaler=scaler,
            threshold=0.5,
        )
        val_threshold_results, epoch_best_threshold = evaluate_thresholds(
            model,
            val_loader,
            criterion,
            device,
            args,
            thresholds,
        )
        val_stats = val_threshold_results[epoch_best_threshold]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_dice": train_stats["dice"],
            "train_pred_pos_ratio": train_stats["pred_pos_ratio"],
            "train_gt_pos_ratio": train_stats["gt_pos_ratio"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_dice": val_stats["dice"],
            "val_pred_pos_ratio": val_stats["pred_pos_ratio"],
            "val_gt_pos_ratio": val_stats["gt_pos_ratio"],
            "val_threshold": epoch_best_threshold,
        }
        history.append(row)

        line = (
            f"epoch={epoch:03d} lr={lr:.6f} "
            f"train_loss={row['train_loss']:.6f} train_iou={row['train_iou']:.6f} train_dice={row['train_dice']:.6f} "
            f"train_pred_pos={row['train_pred_pos_ratio']:.6f} train_gt_pos={row['train_gt_pos_ratio']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_iou={row['val_iou']:.6f} val_dice={row['val_dice']:.6f} "
            f"val_pred_pos={row['val_pred_pos_ratio']:.6f} val_gt_pos={row['val_gt_pos_ratio']:.6f} "
            f"val_thr={row['val_threshold']:.2f}"
        )
        print(line)
        append_log_line(txt_log_path, line)

        if row["val_dice"] > best_dice or train_on_trainval:
            best_dice = row["val_dice"]
            best_threshold = epoch_best_threshold
            if not train_on_trainval:
                no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model, args),
                    "optimizer_state": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_threshold": best_threshold,
                    "no_improve_epochs": no_improve_epochs,
                    "args": vars(args),
                },
                save_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        if epoch % args.save_last_every == 0 or epoch == end_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model, args),
                    "optimizer_state": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_threshold": best_threshold,
                    "no_improve_epochs": no_improve_epochs,
                    "args": vars(args),
                },
                save_dir / "last.pt",
            )

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.save_debug_vis and hasattr(model, "set_debug_capture"):
            debug_dir = save_dir / "debug_vis"
            debug_dir.mkdir(parents=True, exist_ok=True)
            model.set_debug_capture(True)
            debug_batch = next(iter(val_loader))
            with torch.no_grad():
                _mask, _logits, _aux = forward_model(debug_batch, model, args, device)
                debug_payload = model.get_debug_outputs() if hasattr(model, "get_debug_outputs") else {}
            model.set_debug_capture(False)
            torch.save(debug_payload, debug_dir / f"epoch_{epoch:03d}.pt")

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            append_log_line(
                txt_log_path,
                (
                    f"early_stop epoch={epoch} "
                    f"no_improve_epochs={no_improve_epochs} "
                    f"best_dice={best_dice:.6f} best_threshold={best_threshold:.2f}"
                ),
            )
            print(
                f"Early stopping at epoch {epoch}: "
                f"no improvement for {no_improve_epochs} epochs "
                f"(best_dice={best_dice:.4f}, best_threshold={best_threshold:.2f})"
            )
            break

    best_ckpt = load_checkpoint(save_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state"], strict=False)
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))
    use_tta = bool(getattr(args, "use_tta", False))
    if use_tta:
        test_stats = run_test_with_tta(
            model,
            test_loader,
            criterion,
            device,
            args,
            tokenizer,
            threshold=best_threshold,
        )
    else:
        test_stats = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            args,
            optimizer=None,
            scaler=None,
            threshold=best_threshold,
        )

    summary = {
        "best_epoch": int(best_ckpt.get("epoch", -1)),
        "best_threshold": best_threshold,
        "tta": use_tta,
        "loss": float(test_stats["loss"]),
        "iou": float(test_stats["iou"]),
        "dice": float(test_stats["dice"]),
        "per_image_iou": float(test_stats["per_image_iou"]),
        "per_image_dice": float(test_stats["per_image_dice"]),
        "global_iou": float(test_stats["global_iou"]),
        "global_dice": float(test_stats["global_dice"]),
        "num_test_images": int(test_stats["num_images"]),
        "pred_pos_ratio": float(test_stats["pred_pos_ratio"]),
        "gt_pos_ratio": float(test_stats["gt_pos_ratio"]),
    }
    (save_dir / "final_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_test_txt_path.write_text(
        (
            f"best_epoch={summary['best_epoch']} best_threshold={summary['best_threshold']:.2f} "
            f"test_loss={summary['loss']:.6f} test_per_image_iou={summary['per_image_iou']:.6f} "
            f"test_per_image_dice={summary['per_image_dice']:.6f} "
            f"test_global_iou={summary['global_iou']:.6f} test_global_dice={summary['global_dice']:.6f} "
            f"test_pred_pos={summary['pred_pos_ratio']:.6f} test_gt_pos={summary['gt_pos_ratio']:.6f}\n"
        ),
        encoding="utf-8",
    )
    append_log_line(txt_log_path, json.dumps(summary, ensure_ascii=False))
    print(f"Training complete. Final test: {summary}")


if __name__ == "__main__":
    main()
