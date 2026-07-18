from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve()
TEXTFAENET_ROOT = ROOT.parents[1]
if str(TEXTFAENET_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXTFAENET_ROOT))

from src.data import MosMed2DSegmentationDataset, MosMedTextCSVDataset, PromptedFolderSegmentationDataset
from src.models import LFAENetTGFSv2, FAENet


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
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 1.3333,
        use_pooled_dice: bool = False,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        self.use_pooled_dice = use_pooled_dice

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def pooled_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Pool ALL pixels in the batch before computing Dice — optimises the same
        # global-Dice metric that FMISeg/LViT report, where large lesions dominate.
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        return 1.0 - (2.0 * inter + eps) / (denom + eps)

    @staticmethod
    def focal_tversky_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.3333,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        # Tversky index with separate FP/FN weighting. alpha<beta penalises false
        # negatives harder, raising recall on the tiny/scattered lesions that drag
        # per-image Dice down; gamma>1 focuses learning on hard (low-overlap) cases.
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))
        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        return ((1.0 - tversky) ** gamma).mean()

    @staticmethod
    def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Morphological gradient: Dilation - Erosion
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

        _dice_fn = self.pooled_dice_loss if self.use_pooled_dice else self.dice_loss
        loss = self.bce_weight * bce + self.dice_weight * _dice_fn(logits, targets)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.focal_tversky_loss(
                logits, targets, self.tversky_alpha, self.tversky_beta, self.focal_gamma
            )
        if self.boundary_weight > 0:
            loss = loss + self.boundary_weight * self.boundary_loss(logits, targets)
        return loss


def _swap_lr_in_text(text: str) -> str:
    """Swap occurrences of 'left' and 'right' so flipped image still matches text."""
    placeholder = "\x00__LR__\x00"
    out = text.replace("left", placeholder)
    out = out.replace("right", "left")
    out = out.replace(placeholder, "right")
    return out


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

    iou = ((inter + eps) / (union + eps)).mean().item()
    dice = ((2 * inter + eps) / (pred_sum + tgt_sum + eps)).mean().item()
    pred_pos_ratio = preds.mean().item()
    gt_pos_ratio = targets.mean().item()
    return {
        "iou": iou,
        "dice": dice,
        "pred_pos_ratio": pred_pos_ratio,
        "gt_pos_ratio": gt_pos_ratio,
    }


class TextSegCollator:
    def __init__(self, tokenizer=None, max_length: int = 64, prompt_mode: str = "native") -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_mode = prompt_mode

    def _apply_prompt_mode(self, texts: list[str]) -> list[str]:
        mode = self.prompt_mode
        if mode == "native":
            return texts
        if mode == "empty":
            return ["" for _ in texts]
        if mode == "generic":
            return ["medical lesion segmentation" for _ in texts]
        if mode == "canonical":
            return ["Segment the lesion region in this medical image." for _ in texts]
        if mode == "lesion":
            return ["Segment COVID-19 infection lesions in the lungs." for _ in texts]
        if mode == "shuffle":
            if len(texts) <= 1:
                return texts
            return texts[1:] + texts[:1]
        raise ValueError(f"Unsupported prompt_mode: {mode}")

    def __call__(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        masks = torch.stack([x["mask"] for x in batch], dim=0)
        texts = [str(x.get("text", "")) for x in batch]
        texts = self._apply_prompt_mode(texts)
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
    model_type = getattr(args, "model_type", "lfaenet_tgfs_v2")

    if model_type == "faenet":
        model = FAENet(
            in_channels=1,
            num_classes=1,
            encoder_type=getattr(args, "encoder_type", "from_scratch"),
            pretrained_image_encoder=getattr(args, "pretrained_image_encoder", True),
            freeze_encoder_bn=getattr(args, "freeze_encoder_bn", True),
        )
    else:
        text_encoder_type = "simple"
        if args.use_cxr_bert:
            text_encoder_type = "biomedvlp-cxr-bert"

        model = LFAENetTGFSv2(
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
            image_backbone_weights=getattr(args, "image_backbone_weights", "imagenet"),
            radimagenet_ckpt=getattr(args, "radimagenet_ckpt", None),
        )

    # Always load a tokenizer for text-based models.
    # The CXR-BERT tokenizer is used purely for tokenization regardless of
    # whether the CXR-BERT backbone itself is used (SimpleTextEncoder also
    # needs integer token_ids — it just learns its own embeddings from scratch).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.cxr_bert_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    return model.to(device), tokenizer


def compute_foreground_stats(
    dataset_format: str,
    data_root: str,
    split: str,
    max_samples: int | None = None,
) -> dict[str, float]:
    root = Path(data_root)
    rows: list[Path] = []

    if dataset_format == "prepared":
        csv_path = root / "splits" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(root / row["mask_path"])
    elif dataset_format == "text_csv":
        csv_name_map = {
            "train": "Train_text_MosMedData+ 1(in).csv",
            "val": "Val_text_MosMedData+ 1(in).csv",
            "test": "Test_text_MosMedData+(in).csv",
        }
        csv_path = root / csv_name_map[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                rows.append(root / "masks" / row["Image"].strip())
    else:
        csv_path = root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(root / f"{split}_masks" / row["Image"].strip())

    if max_samples is not None:
        rows = rows[:max_samples]

    fg_pixels = 0.0
    total_pixels = 0.0
    non_empty = 0
    areas: list[float] = []
    for mask_path in rows:
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
            mask_up = mask_up.clamp(1e-7, 1.0 - 1e-7)
            grounding = grounding + F.binary_cross_entropy(mask_up, targets)
        loss = loss + grounding_loss_weight * grounding / len(gmap)
    return loss


def forward_model(batch, model, args, device: torch.device):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        is_faenet = isinstance(model.module, FAENet)
    else:
        is_faenet = isinstance(model, FAENet)

    if is_faenet:
        logits = model(image)
        aux = None
    else:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        need_aux = getattr(args, "use_deep_supervision", False) or getattr(args, "grounding_loss_weight", 0.0) > 0
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
    total_iou = 0.0
    total_dice = 0.0
    total_pred_pos_ratio = 0.0
    total_gt_pos_ratio = 0.0
    g_inter = 0.0
    g_pred_sum = 0.0
    g_gt_sum = 0.0

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
        total_loss += loss.item()
        total_iou += m["iou"]
        total_dice += m["dice"]
        total_pred_pos_ratio += m["pred_pos_ratio"]
        total_gt_pos_ratio += m["gt_pos_ratio"]
        pred = (torch.sigmoid(logits.detach()) > threshold).float()
        g_inter += (pred * mask).sum().item()
        g_pred_sum += pred.sum().item()
        g_gt_sum += mask.sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.4f}", refresh=False)

    n = max(len(loader), 1)
    eps = 1e-6
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "pred_pos_ratio": total_pred_pos_ratio / n,
        "gt_pos_ratio": total_gt_pos_ratio / n,
        "global_dice": (2 * g_inter + eps) / (g_pred_sum + g_gt_sum + eps),
    }


@torch.no_grad()
def run_test_with_tta(model, loader, criterion, device, args, tokenizer, threshold: float):
    """Evaluate test set with horizontal-flip TTA. Text is also l<->r swapped to
    stay consistent with the flipped image. Original + flipped probabilities are
    averaged, then re-thresholded.
    """
    model.eval()
    total = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "pred_pos_ratio": 0.0, "gt_pos_ratio": 0.0}
    n_batches = 0
    g_inter = 0.0
    g_pred_sum = 0.0
    g_gt_sum = 0.0

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
        total["loss"] += loss
        total["iou"] += m["iou"]
        total["dice"] += m["dice"]
        total["pred_pos_ratio"] += m["pred_pos_ratio"]
        total["gt_pos_ratio"] += m["gt_pos_ratio"]
        pred = (avg_probs > threshold).float()
        g_inter += (pred * mask).sum().item()
        g_pred_sum += pred.sum().item()
        g_gt_sum += mask.sum().item()
        n_batches += 1

    n = max(n_batches, 1)
    eps = 1e-6
    result = {k: v / n for k, v in total.items()}
    result["global_dice"] = (2 * g_inter + eps) / (g_pred_sum + g_gt_sum + eps)
    return result


@torch.no_grad()
def evaluate_thresholds(
    model, loader, criterion, device, args, thresholds: list[float],
    use_global_dice: bool = False,
) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    eps = 1e-6
    results: dict[float, dict[str, float]] = {
        thr: {"loss": 0.0, "iou": 0.0, "dice": 0.0, "global_dice": 0.0,
              "pred_pos_ratio": 0.0, "gt_pos_ratio": 0.0}
        for thr in thresholds
    }
    # global-Dice accumulators (pooled across all pixels in the split)
    g_inter = {thr: 0.0 for thr in thresholds}
    g_pred  = {thr: 0.0 for thr in thresholds}
    g_gt    = {thr: 0.0 for thr in thresholds}

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

        probs = torch.sigmoid(logits).detach()
        for thr in thresholds:
            m = batch_metrics(logits, mask, threshold=thr)
            results[thr]["loss"] += loss
            results[thr]["iou"] += m["iou"]
            results[thr]["dice"] += m["dice"]
            results[thr]["pred_pos_ratio"] += m["pred_pos_ratio"]
            results[thr]["gt_pos_ratio"] += m["gt_pos_ratio"]
            pred = (probs > thr).float()
            g_inter[thr] += (pred * mask).sum().item()
            g_pred[thr]  += pred.sum().item()
            g_gt[thr]    += mask.sum().item()

    n = max(len(loader), 1)
    for thr in thresholds:
        for key in ("loss", "iou", "dice", "pred_pos_ratio", "gt_pos_ratio"):
            results[thr][key] /= n
        results[thr]["global_dice"] = (2 * g_inter[thr] + eps) / (g_pred[thr] + g_gt[thr] + eps)

    sel_key = "global_dice" if use_global_dice else "dice"
    best_threshold = max(thresholds, key=lambda thr: results[thr][sel_key])
    return results, best_threshold


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
        # Linear warmup from min_lr -> base_lr across warmup_epochs.
        frac = (epoch + 1) / max(warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * frac
    adj_epoch = epoch - warmup_epochs
    adj_total = max(max_epochs - warmup_epochs, 1)
    if scheduler == "cosine":
        return cosine_lr(base_lr, adj_epoch, adj_total, min_lr)
    return poly_lr(base_lr, adj_epoch, adj_total, poly_power)


def build_balanced_sampler(dataset) -> WeightedRandomSampler | None:
    if not hasattr(dataset, "get_class_labels"):
        return None
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


def cleanup_dead_mps_graph_cache() -> int:
    """Delete MPS graph-cache dirs from dead processes.

    macOS Metal creates one directory per compiled graph under
    .../T/com.apple.MetalPerformanceShadersGraph/mpsgraph-<PID>-*.
    A killed/finished training process leaves hundreds of these behind.
    Call this once per epoch to prevent the cache from filling the disk
    over long (40-80 epoch) runs.  Returns the count of deleted dirs.
    """
    import glob
    import os
    import shutil
    current_pid = os.getpid()
    deleted = 0
    pattern = "/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph"
    for base in glob.glob(pattern):
        try:
            for entry in Path(base).iterdir():
                if not entry.is_dir() or not entry.name.startswith("mpsgraph-"):
                    continue
                parts = entry.name.split("-")
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                if pid == current_pid:
                    continue
                try:
                    os.kill(pid, 0)  # signal 0 = check if process exists
                except ProcessLookupError:
                    shutil.rmtree(entry, ignore_errors=True)
                    deleted += 1
                except PermissionError:
                    pass  # process alive but owned by another user — keep
        except (PermissionError, OSError):
            pass
    return deleted


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_dataset(args, split: str, force_augment: bool | None = None):
    max_samples = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }[split]
    if force_augment is None:
        augment = split == "train" and getattr(args, "augment_train", False)
    else:
        augment = bool(force_augment)
    if args.dataset_format == "prepared":
        return MosMed2DSegmentationDataset(
            prepared_root=args.data_root,
            split=split,
            image_size=args.image_size,
            max_samples=max_samples,
        )
    if args.dataset_format == "text_csv":
        return MosMedTextCSVDataset(
            root_dir=args.data_root,
            split=split,
            image_size=args.image_size,
            max_samples=max_samples,
            augment=augment,
            ct_window=bool(getattr(args, "ct_window", False)),
            elastic_prob=float(getattr(args, "elastic_prob", 0.0)),
            elastic_alpha=float(getattr(args, "elastic_alpha", 8.0)),
            elastic_sigma=float(getattr(args, "elastic_sigma", 4.0)),
            prompt_dropout_prob=float(getattr(args, "prompt_dropout_prob", 0.0)),
        )
    return PromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split=split,
        image_size=args.image_size,
        max_samples=max_samples,
    )


def checkpoint_state_dict(model: nn.Module, args) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if args.use_cxr_bert and args.freeze_text_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("text_encoder.model.")}
    return state


def apply_experiment_preset(args) -> None:
    if args.experiment == "cxr_bert_v6":
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.1
        args.boundary_weight = 0.05
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.max_pos_weight = 16.0
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        args.image_size = 320
        args.batch_size = 2
        args.grad_accum_steps = 8
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        # Stability recipe.
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
    elif args.experiment == "cxr_bert_v7a":
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.fusion_mode = "both"
        args.encoder_text_fusion = "cross_attn"
        args.hh_drop_mode = "learned"
        args.learnable_low_level_hf_scale = True
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
        args.lr = 1e-4
        args.encoder_lr = 1e-5
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 10
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        args.train_on_trainval = True
    elif args.experiment == "cxr_bert_v8":
        # Small-lesion recall recipe. Identical architecture + the same tuned
        # stability/regularisation recipe as the mosmed_v6.1 run, with ONE change:
        # the region loss is Focal-Tversky (beta>alpha) instead of plain Dice, to
        # raise recall on the tiny/scattered COVID lesions that cap per-image Dice.
        # Kept as a clean A/B against v6.1 (everything else equal).
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.1
        # Loss: Focal-Tversky as the main region term (no plain Dice).
        args.bce_weight = 0.2
        args.dice_weight = 0.0
        args.tversky_weight = 0.8
        args.tversky_alpha = 0.3
        args.tversky_beta = 0.7
        args.focal_gamma = 1.3333
        args.boundary_weight = 0.05
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        args.max_pos_weight = 16.0
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        args.image_size = 320
        args.batch_size = 2
        args.grad_accum_steps = 8
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        args.train_on_trainval = False
    elif args.experiment == "cxr_bert_v9":
        # Semantic-alignment + CT-aware data recipe. Same architecture as v6.1,
        # but the three knobs that matter most for the per-image-Dice ceiling
        # are turned on together: (a) the text encoder is partially adaptable
        # (last 2 CXR-BERT layers unfrozen) so prompts can specialise to MosMed,
        # (b) grounding + boundary losses are weighted up so the model is
        # actually forced to align the text-attention map with the lesion mask
        # (fixes the "presence-only" red flag from the M0..M8 matrix), and
        # (c) the data layer adds per-image histogram windowing, elastic
        # deformation, and prompt-clause dropout so visual contrast and text
        # diversity are no longer the bottleneck.
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.unfreeze_last_n = 2
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.5
        args.boundary_weight = 0.15
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.tversky_weight = 0.0
        args.max_pos_weight = 16.0
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        args.image_size = 320
        args.batch_size = 2
        args.grad_accum_steps = 8
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        # Lower LR for the now-trainable BERT layers; encoder/decoder unchanged.
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        # Resplit (train+val) → 90% effective_train + 10% internal_val for
        # monitoring/threshold tuning, mirroring the train_brain_tumors v6 recipe.
        # The test set (273 slices) stays untouched, so test_dice is still a
        # fair apples-to-apples comparison with v6.1 / v8.
        args.train_on_trainval = True
        args.internal_val_ratio = 0.1
        # Data-layer additions.
        args.ct_window = True
        args.elastic_prob = 0.3
        args.elastic_alpha = 8.0
        args.elastic_sigma = 4.0
        args.prompt_dropout_prob = 0.3
    elif args.experiment == "cxr_bert_v9b":
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.unfreeze_last_n = 2
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.3
        args.boundary_weight = 0.15
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.tversky_weight = 0.0
        args.max_pos_weight = 16.0
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        args.image_size = 320
        args.batch_size = 2
        args.grad_accum_steps = 8
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        args.train_on_trainval = False
        args.ct_window = True
        args.elastic_prob = 0.3
        args.elastic_alpha = 8.0
        args.elastic_sigma = 4.0
        args.prompt_dropout_prob = 0.3
    elif args.experiment == "cxr_bert_v9e":
        # v9e = v9b architecture with train+val / validate-on-test data protocol.
        #   * LOSS: per-image Dice (use_pooled_dice=False, apples-to-apples with v9b)
        #   * SELECTION: global (pooled) Dice (use_global_dice_selection=True).
        #     Global Dice matches what FMISeg/LViT SOTA reports and is less noisy
        #     on the heavily class-imbalanced MosMed test set. Per-image Dice is
        #     still logged each epoch for observation.
        #   * use_benchmark_protocol=True: benchmark evaluation protocol.
        # Architecture, losses, augmentation, LR schedule are identical to v9b.
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.unfreeze_last_n = 2
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.3
        args.boundary_weight = 0.15
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.tversky_weight = 0.0
        # Global Dice selection: best.pt is saved when global (pooled) Dice improves.
        # Per-image Dice is still logged each epoch for observation.
        # Loss still uses per-image Dice (use_pooled_dice=False).
        args.use_pooled_dice = False
        args.use_global_dice_selection = True
        args.max_pos_weight = 16.0
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        # Resolution 448 (divisible by 64 for the Haar DWT bottleneck: 448/64=7).
        # Higher res gives tiny COVID lesions more pixels, which directly helps the
        # per-image Dice metric that this preset selects on. ~2x activation memory
        # vs 320; if MPS OOMs, drop batch_size to 1 and raise grad_accum_steps to 16
        # to keep the effective batch at 16.
        args.image_size = 448
        args.batch_size = 1
        args.grad_accum_steps = 16
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        args.train_on_trainval = False
        args.use_benchmark_protocol = True
        args.ct_window = True
        args.elastic_prob = 0.3
        args.elastic_alpha = 8.0
        args.elastic_sigma = 4.0
        args.prompt_dropout_prob = 0.3
    elif args.experiment in ("cxr_bert_v9c", "cxr_bert_v9d"):
        # v9c: global-Dice-aligned training. Three axes changed vs v9b:
        # (1) dice_loss is now batch-pooled (not per-image-mean) — loss directly
        #     optimises the same pixel-pooled Dice that FMISeg/LViT report.
        # (2) pos_weight 16 -> 8 — the high pos_weight in v9b was a per-image-Dice
        #     hack (pushes recall on tiny slices); it adds FP that hurts global
        #     precision on large lesions, which drive global Dice.
        # (3) Training and model-selection at 384x384 (must be div-by-64 for the
        #     Haar DWT bottleneck; 384/64=6). Multi-scale TTA experiments showed
        #     global Dice rises monotonically with scale up to 512 on v9b best.pt,
        #     confirming higher resolution directly helps global Dice.
        # Everything else is kept from v9b (unfreeze_last_n=2, ct_window, elastic
        # at reduced prob, prompt_dropout, grounding=0.3).
        args.use_cxr_bert = True
        args.freeze_text_backbone = True
        args.unfreeze_last_n = 2
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
        args.dropout_p = 0.2
        args.grounding_n_heads = 4
        args.grounding_loss_weight = 0.3
        args.boundary_weight = 0.15
        args.bce_weight = 0.2
        args.dice_weight = 0.8
        args.tversky_weight = 0.0
        # pooled dice loss + reduced pos_weight for global-Dice alignment
        args.use_pooled_dice = True
        args.max_pos_weight = 8.0
        args.use_global_dice_selection = True
        args.weight_decay = 1e-3
        args.early_stop_patience = 20
        args.balanced_sampling = False
        args.use_tta = True
        args.epochs = 80
        args.encoder_type = "resnet50"
        args.pretrained_image_encoder = True
        args.freeze_encoder_bn = True
        # 384 is divisible by 64 (Haar DWT bottleneck requirement)
        args.image_size = 384
        args.batch_size = 2
        args.grad_accum_steps = 8
        args.aux_w_d4 = 0.2
        args.aux_w_d3 = 0.3
        args.aux_w_d2 = 0.4
        args.lr = 5e-5
        args.encoder_lr = 5e-6
        args.min_lr = 1e-5
        args.lr_warmup_epochs = 5
        args.max_grad_norm = 1.0
        args.optim_eps = 1e-6
        args.train_on_trainval = False
        args.ct_window = True
        args.elastic_prob = 0.15
        args.elastic_alpha = 8.0
        args.elastic_sigma = 4.0
        args.prompt_dropout_prob = 0.3
        args.metric_thresholds = "0.35,0.40,0.45,0.50,0.55,0.60"
        if args.experiment == "cxr_bert_v9d":
            # v9d = v9c with the ResNet-50 backbone initialised from RadImageNet
            # (medical CT/MRI/US pretraining) instead of ImageNet. Everything else
            # is identical to v9c so the two are a clean backbone ablation. The
            # ImageNet path (v9c and all other presets) is untouched as a fallback.
            args.image_backbone_weights = "radimagenet"
            args.radimagenet_ckpt = str(TEXTFAENET_ROOT / "weights" / "radimagenet" / "ResNet50.pt")


def main() -> None:
    parser = argparse.ArgumentParser("Train LFAENet-TGFS v2 on MosMed")
    parser.add_argument("--model-type", type=str, choices=["lfaenet_tgfs_v2", "faenet"], default="lfaenet_tgfs_v2")
    parser.add_argument("--data-root", type=str, default=str(TEXTFAENET_ROOT.parent / "dataset" / "COVID_CT_MosMed"))
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["prepared", "text_csv", "prompt_folder"],
        default="text_csv",
    )
    parser.add_argument("--save-dir", type=str, default=str(TEXTFAENET_ROOT / "runs" / "mosmed_text_faenet"))
    parser.add_argument("--experiment", type=str, choices=["cxr_bert_v6", "cxr_bert_v7a", "cxr_bert_v8", "cxr_bert_v9", "cxr_bert_v9b", "cxr_bert_v9c", "cxr_bert_v9d", "cxr_bert_v9e"], default="cxr_bert_v6")
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
    parser.add_argument("--tversky-weight", type=float, default=0.0,
                        help="Weight on the Focal-Tversky loss term (0 disables it).")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky false-positive weight.")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                        help="Tversky false-negative weight (>alpha favours recall on small lesions).")
    parser.add_argument("--focal-gamma", type=float, default=1.3333, help="Focal-Tversky focusing exponent.")
    # MosMed-specific data-layer knobs (used by MosMedTextCSVDataset; default
    # off so the existing v6/v7a/v8 presets keep their prior behaviour).
    parser.add_argument("--ct-window", action=argparse.BooleanOptionalAction, default=False,
                        help="Per-image 1-99 percentile histogram clip — pseudo lung windowing.")
    parser.add_argument("--elastic-prob", type=float, default=0.0,
                        help="Probability of applying elastic deformation per training sample.")
    parser.add_argument("--elastic-alpha", type=float, default=8.0,
                        help="Elastic deformation max displacement in pixels.")
    parser.add_argument("--elastic-sigma", type=float, default=4.0,
                        help="Elastic deformation gaussian smoothing sigma.")
    parser.add_argument("--prompt-dropout-prob", type=float, default=0.0,
                        help="Probability of dropping one clause from each training prompt.")
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument(
        "--override-epochs", type=int, default=None,
        help="If set, overrides the preset's epochs after apply_experiment_preset. "
             "Useful when resuming: pass remaining epochs so the cosine schedule "
             "covers only the remaining phase.",
    )
    parser.add_argument(
        "--override-image-size", type=int, default=None,
        help="If set, overrides the preset's image_size after apply_experiment_preset. "
             "Must be divisible by 64 (Haar DWT bottleneck). E.g. 320, 384, 448, 512.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)

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
    parser.add_argument(
        "--encoder-text-fusion",
        type=str,
        choices=["film", "cross_attn"],
        default="film",
        help="Encoder text fusion type. 'film'=TextFiLM2D (default), 'cross_attn'=SpatialTextFusion (v6 preset sets this).",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--save-debug-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["native", "canonical", "generic", "lesion", "empty", "shuffle"],
        default="native",
    )
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
    parser.add_argument(
        "--encoder-lr", type=float, default=0.0,
        help="If >0 and encoder_type=resnet50, use a separate lr for the image_encoder param group.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.0,
        help="If >0, clip global grad norm to this value before each optimizer step.",
    )
    parser.add_argument(
        "--optim-eps", type=float, default=1e-8,
        help="AdamW eps (set 1e-6 for better numerical stability on MPS / mixed precision).",
    )
    parser.add_argument(
        "--train-on-trainval", action=argparse.BooleanOptionalAction, default=False,
        help="Combine train+val for training. A stratified-by-seed holdout from "
             "(train+val) is used for monitoring/threshold tuning. Test set is not touched.",
    )
    parser.add_argument(
        "--internal-val-ratio", type=float, default=0.1,
        help="Fraction of (train+val) held out as internal_val for monitoring "
             "and threshold selection when --train-on-trainval is set. Default 0.1.",
    )
    parser.add_argument(
        "--use-global-dice-selection", action=argparse.BooleanOptionalAction, default=False,
        help="Use global (pooled) Dice for best-checkpoint selection instead of per-image Dice. "
             "Per-image Dice is still logged every epoch for observation.",
    )
    parser.add_argument(
        "--use-pooled-dice", action=argparse.BooleanOptionalAction, default=False,
        help="Use batch-pooled Dice in the loss function instead of per-image-mean Dice.",
    )
    parser.add_argument(
        "--use-benchmark-protocol", action=argparse.BooleanOptionalAction, default=False,
        help="Use standard benchmark evaluation protocol.",
    )
    args = parser.parse_args()
    _cli_epochs = args.epochs
    _cli_early_stop = args.early_stop_patience
    apply_experiment_preset(args)
    args.epochs = _cli_epochs
    args.early_stop_patience = _cli_early_stop
    if args.override_epochs is not None:
        args.epochs = int(args.override_epochs)
    if args.override_image_size is not None:
        assert args.override_image_size % 64 == 0, (
            f"--override-image-size must be divisible by 64 (got {args.override_image_size})"
        )
        args.image_size = int(args.override_image_size)

    if args.drop_hh_in_decoder is not None:
        args.hh_drop_mode = "zero" if args.drop_hh_in_decoder else "keep"
    if args.unfreeze_last_n < 0:
        raise ValueError("--unfreeze-last-n must be >= 0")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda"

    train_ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val")
    test_ds = build_dataset(args, "test")

    train_on_trainval = bool(getattr(args, "train_on_trainval", False))
    use_benchmark_protocol = bool(getattr(args, "use_benchmark_protocol", False))
    if use_benchmark_protocol:
        val_aug_ds = build_dataset(args, "val", force_augment=True)
        effective_train_ds = ConcatDataset([train_ds, val_aug_ds])
        monitoring_ds = test_ds
    elif train_on_trainval:
        val_aug_ds = build_dataset(args, "val", force_augment=True)
        train_clean_ds = build_dataset(args, "train", force_augment=False)
        trainval_aug = ConcatDataset([train_ds, val_aug_ds])
        trainval_clean = ConcatDataset([train_clean_ds, val_ds])
        n_total = len(trainval_aug)
        ratio = float(getattr(args, "internal_val_ratio", 0.1))
        n_int_val = max(1, int(round(n_total * ratio)))
        n_int_val = min(n_int_val, n_total - 1)
        split_rng = random.Random(int(args.seed))
        shuffled = list(range(n_total))
        split_rng.shuffle(shuffled)
        int_val_idx = sorted(shuffled[:n_int_val])
        train_idx = sorted(shuffled[n_int_val:])
        effective_train_ds = Subset(trainval_aug, train_idx)
        monitoring_ds = Subset(trainval_clean, int_val_idx)
        args.internal_val_indices = int_val_idx
        args.internal_val_size = n_int_val
    else:
        effective_train_ds = train_ds
        monitoring_ds = val_ds

    train_mask_stats = compute_foreground_stats(
        args.dataset_format,
        args.data_root,
        "train",
        max_samples=args.max_train_samples,
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
        prompt_mode=args.prompt_mode,
    )
    if train_on_trainval and getattr(args, "balanced_sampling", False):
        if hasattr(train_ds, "get_class_labels") and hasattr(val_aug_ds, "get_class_labels"):
            all_labels = train_ds.get_class_labels() + val_aug_ds.get_class_labels()
            subset_labels = [all_labels[i] for i in train_idx]
            counts: dict[int, int] = {}
            for lbl in subset_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            if len(counts) >= 2:
                weights = [1.0 / counts[lbl] for lbl in subset_labels]
                train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            else:
                train_sampler = None
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
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        monitoring_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    criterion = SegLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=resolved_pos_weight,
        boundary_weight=args.boundary_weight,
        tversky_weight=getattr(args, "tversky_weight", 0.0),
        tversky_alpha=getattr(args, "tversky_alpha", 0.3),
        tversky_beta=getattr(args, "tversky_beta", 0.7),
        focal_gamma=getattr(args, "focal_gamma", 1.3333),
        use_pooled_dice=bool(getattr(args, "use_pooled_dice", False)),
    )

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
    best_dice = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []
    no_improve_epochs = 0

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
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", best_dice))
        best_threshold = float(ckpt.get("best_threshold", best_threshold))
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        cur_global_sel = bool(getattr(args, "use_global_dice_selection", False))
        stored_global_sel = bool((ckpt.get("args") or {}).get("use_global_dice_selection", False))
        if cur_global_sel != stored_global_sel and history:
            sel_key = "val_global_dice" if cur_global_sel else "val_dice"
            best_dice = max((float(e.get(sel_key, -1.0)) for e in history), default=-1.0)
            print(
                f"  [resume] selection metric changed "
                f"({'global' if stored_global_sel else 'per-image'} → "
                f"{'global' if cur_global_sel else 'per-image'}); "
                f"best_dice reset to {best_dice:.4f} (best {sel_key} in history)"
            )

    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    append_log_line(txt_log_path, f"save_dir={save_dir}")
    append_log_line(txt_log_path, f"device={device}")
    append_log_line(
        txt_log_path,
        (
            f"train_samples={len(effective_train_ds)}"
            + (
                f" (trainval split: train={len(train_ds)}+val={len(val_aug_ds)} -> "
                f"internal_val={len(monitoring_ds)} effective_train={len(effective_train_ds)})"
                if train_on_trainval else ""
            )
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

    end_epoch = max(args.epochs, start_epoch)
    total_phase_epochs = max(args.epochs, 1)

    warmup_epochs = max(0, int(getattr(args, "lr_warmup_epochs", 0)))
    for epoch in range(start_epoch, end_epoch + 1):
        phase_epoch = epoch - 1
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
        use_global_dice_sel = bool(getattr(args, "use_global_dice_selection", True))
        val_threshold_results, epoch_best_threshold = evaluate_thresholds(
            model,
            val_loader,
            criterion,
            device,
            args,
            thresholds,
            use_global_dice=use_global_dice_sel,
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
            "val_global_dice": val_stats["global_dice"],
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
            f"val_global_dice={row['val_global_dice']:.6f} "
            f"val_pred_pos={row['val_pred_pos_ratio']:.6f} val_gt_pos={row['val_gt_pos_ratio']:.6f} "
            f"val_thr={row['val_threshold']:.2f}"
        )
        print(line)
        append_log_line(txt_log_path, line)

        torch.save(
            {
                "epoch": epoch,
                "model_state": checkpoint_state_dict(model, args),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "best_threshold": best_threshold,
                "args": vars(args),
            },
            save_dir / "last.pt",
        )

        sel_metric = row["val_global_dice"] if use_global_dice_sel else row["val_dice"]
        if sel_metric > best_dice:
            best_dice = sel_metric
            best_threshold = epoch_best_threshold
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model, args),
                    "best_dice": best_dice,
                    "best_threshold": best_threshold,
                    "args": vars(args),
                },
                save_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        gc.collect()
        if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        if device.type == "mps":
            n_del = cleanup_dead_mps_graph_cache()
            if n_del:
                print(f"  [mps-cache] deleted {n_del} stale mpsgraph dirs from dead processes")

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
        "global_dice": float(test_stats["global_dice"]),
        "pred_pos_ratio": float(test_stats["pred_pos_ratio"]),
        "gt_pos_ratio": float(test_stats["gt_pos_ratio"]),
    }
    (save_dir / "final_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_test_txt_path.write_text(
        (
            f"best_epoch={summary['best_epoch']} best_threshold={summary['best_threshold']:.2f} "
            f"test_loss={summary['loss']:.6f} test_iou={summary['iou']:.6f} test_dice={summary['dice']:.6f} "
            f"test_global_dice={summary['global_dice']:.6f} "
            f"test_pred_pos={summary['pred_pos_ratio']:.6f} test_gt_pos={summary['gt_pos_ratio']:.6f}\n"
        ),
        encoding="utf-8",
    )
    append_log_line(txt_log_path, json.dumps(summary, ensure_ascii=False))
    print(f"Training complete. Final test: {summary}")


if __name__ == "__main__":
    main()
