from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import FAENet, LFAENetTGFS, LFAENetTGFSv2

PROMPT_MODE_CHOICES = ("native", "canonical", "generic", "lesion", "empty", "shuffle")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        d = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * d


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
    return {"iou": iou, "dice": dice}


def parse_thresholds(spec: str) -> list[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    values = [x for x in values if 0.0 < x < 1.0]
    if not values:
        raise ValueError("No valid thresholds parsed; expected comma-separated values in (0,1).")
    return values


def stable_token_id(token: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    value = 0
    for idx, ch in enumerate(token):
        value = (value + (idx + 1) * ord(ch)) % (vocab_size - 1)
    return value + 1


def simple_tokenize(texts: list[str], max_length: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
    attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
    for row_idx, text in enumerate(texts):
        tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
        ids = [stable_token_id(tok, vocab_size) for tok in tokens[:max_length]]
        if ids:
            input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, : len(ids)] = 1
    return input_ids, attention_mask


def apply_prompt_mode(texts: list[str], mode: str, rng: random.Random) -> list[str]:
    if mode == "native":
        return texts
    if mode == "canonical":
        return [(" ".join(text.strip().lower().split()).rstrip(".") + ".") if text.strip() else "" for text in texts]
    if mode == "generic":
        return ["segment the abnormal medical region." for _ in texts]
    if mode == "lesion":
        return ["segment the lesion region." for _ in texts]
    if mode == "empty":
        return ["" for _ in texts]
    if mode == "shuffle":
        shuffled = list(texts)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unsupported prompt_mode: {mode}")


class TextSegCollator:
    def __init__(
        self,
        tokenizer=None,
        max_length: int = 64,
        prompt_mode: str = "native",
        seed: int = 42,
        simple_vocab_size: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_mode = prompt_mode
        self.rng = random.Random(seed)
        self.simple_vocab_size = simple_vocab_size

    def __call__(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        masks = torch.stack([x["mask"] for x in batch], dim=0)
        texts = apply_prompt_mode([str(x.get("text", "")) for x in batch], self.prompt_mode, self.rng)
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
        elif self.simple_vocab_size is not None:
            input_ids, attention_mask = simple_tokenize(texts, self.max_length, self.simple_vocab_size)
            out["input_ids"] = input_ids
            out["attention_mask"] = attention_mask

        return out


def create_model(args, device: torch.device):
    tokenizer = None
    text_encoder_type = "simple"
    if args.use_cxr_bert:
        text_encoder_type = "biomedvlp-cxr-bert"

    if args.model_type == "faenet":
        model = FAENet(in_channels=1, num_classes=1)
    elif args.model_type == "lfaenet_tgfs":
        model = LFAENetTGFS(
            in_channels=1,
            num_classes=1,
            text_dim=args.text_dim,
            vocab_size=args.vocab_size,
            text_encoder_type=text_encoder_type,
            text_backbone_path=args.cxr_bert_dir,
            freeze_text_backbone=args.freeze_text_backbone,
        )
    else:
        model = LFAENetTGFSv2(
            in_channels=1,
            num_classes=1,
            text_dim=args.text_dim,
            vocab_size=args.vocab_size,
            text_encoder_type=text_encoder_type,
            text_backbone_path=args.cxr_bert_dir,
            freeze_text_backbone=args.freeze_text_backbone,
            drop_hh_in_decoder=args.drop_hh_in_decoder,
            hh_drop_mode=args.hh_drop_mode,
            low_level_hf_scale=args.low_level_hf_scale,
            spatial_sharpen_power=args.spatial_sharpen_power,
            use_deep_supervision=args.use_deep_supervision,
            fusion_mode=args.fusion_mode,
            unfreeze_last_n=args.unfreeze_last_n,
            lora_r=args.lora_r,
        )

    if args.model_type != "faenet" and args.use_cxr_bert:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.cxr_bert_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    return model.to(device), tokenizer


def compute_loss_with_aux(
    criterion,
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: dict[str, torch.Tensor] | None,
    aux_w_d4: float,
    aux_w_d3: float,
    aux_w_d2: float,
) -> torch.Tensor:
    loss = criterion(logits, targets)
    if aux is None:
        return loss
    loss = loss + aux_w_d4 * criterion(aux["d4"], targets)
    loss = loss + aux_w_d3 * criterion(aux["d3"], targets)
    loss = loss + aux_w_d2 * criterion(aux["d2"], targets)
    return loss


def forward_model(batch, model, args, device: torch.device):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    if args.model_type == "faenet":
        logits = model(image)
        aux = None
    else:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        if args.model_type == "lfaenet_tgfs_v2" and args.use_deep_supervision:
            logits, aux = model(image, token_ids=input_ids, attention_mask=attention_mask, return_aux=True)
        else:
            logits = model(image, token_ids=input_ids, attention_mask=attention_mask)
            aux = None

    return image, mask, logits, aux


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, args):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            _, mask, logits, aux = forward_model(batch, model, args, device)
        loss = compute_loss_with_aux(
            criterion,
            logits.float(),
            mask.float(),
            None if aux is None else {k: v.float() for k, v in aux.items()},
            args.aux_w_d4,
            args.aux_w_d3,
            args.aux_w_d2,
        )
        if args.abort_on_nonfinite and (not torch.isfinite(logits).all() or not torch.isfinite(loss)):
            names = batch.get("mask_name", [])
            raise FloatingPointError(f"Non-finite logits/loss detected in batch: {names}")

        loss_for_backward = loss / grad_accum_steps
        if args.use_amp:
            scaler.scale(loss_for_backward).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        m = batch_metrics(logits.detach(), mask)
        total_loss += loss.item()
        total_iou += m["iou"]
        total_dice += m["dice"]

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, args, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            _, mask, logits, aux = forward_model(batch, model, args, device)

        loss = compute_loss_with_aux(
            criterion,
            logits.float(),
            mask.float(),
            None if aux is None else {k: v.float() for k, v in aux.items()},
            args.aux_w_d4,
            args.aux_w_d3,
            args.aux_w_d2,
        )
        if args.abort_on_nonfinite and (not torch.isfinite(logits).all() or not torch.isfinite(loss)):
            names = batch.get("mask_name", [])
            raise FloatingPointError(f"Non-finite logits/loss detected during validation: {names}")
        m = batch_metrics(logits, mask, threshold=threshold)

        total_loss += loss.item()
        total_iou += m["iou"]
        total_dice += m["dice"]

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
    }


@torch.no_grad()
def evaluate_thresholds(model, loader, criterion, device, args, thresholds: list[float]) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    results: dict[float, dict[str, float]] = {
        thr: {"loss": 0.0, "iou": 0.0, "dice": 0.0}
        for thr in thresholds
    }

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            _, mask, logits, aux = forward_model(batch, model, args, device)
        loss = compute_loss_with_aux(
            criterion,
            logits.float(),
            mask.float(),
            None if aux is None else {k: v.float() for k, v in aux.items()},
            args.aux_w_d4,
            args.aux_w_d3,
            args.aux_w_d2,
        ).item()
        if args.abort_on_nonfinite and (not torch.isfinite(logits).all() or not np.isfinite(loss)):
            names = batch.get("mask_name", [])
            raise FloatingPointError(f"Non-finite logits/loss detected during validation: {names}")
        for thr in thresholds:
            m = batch_metrics(logits, mask, threshold=thr)
            results[thr]["loss"] += loss
            results[thr]["iou"] += m["iou"]
            results[thr]["dice"] += m["dice"]

    n = max(len(loader), 1)
    for thr in thresholds:
        for key in results[thr]:
            results[thr][key] /= n
    best_threshold = max(thresholds, key=lambda thr: results[thr]["dice"])
    return results, best_threshold


def poly_lr(base_lr: float, epoch: int, max_epochs: int, power: float) -> float:
    return base_lr * ((1.0 - (epoch / max_epochs)) ** power)


def cosine_lr(base_lr: float, epoch: int, max_epochs: int, min_lr: float) -> float:
    if max_epochs <= 1:
        return base_lr
    t = epoch / (max_epochs - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t))


def append_log_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_checkpoint(path: Path, device: torch.device):
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoints include optimizer state and metadata, so we explicitly disable it.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state_dict(model: nn.Module, args) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    skip_frozen_text = (
        args.model_type in {"lfaenet_tgfs", "lfaenet_tgfs_v2"}
        and args.use_cxr_bert
        and args.freeze_text_backbone
        and int(getattr(args, "unfreeze_last_n", 0)) == 0
        and int(getattr(args, "lora_r", 0)) == 0
    )
    if skip_frozen_text:
        state = {k: v for k, v in state.items() if not k.startswith("text_encoder.model.")}
    return state


def build_checkpoint(
    model: nn.Module,
    optimizer,
    args,
    epoch: int,
    best_dice: float,
    best_threshold: float,
    include_optimizer: bool,
) -> dict:
    ckpt = {
        "epoch": epoch,
        "model_state": checkpoint_state_dict(model, args),
        "args": vars(args),
        "best_dice": best_dice,
        "best_threshold": best_threshold,
    }
    if include_optimizer:
        ckpt["optimizer_state"] = optimizer.state_dict()
    return ckpt


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - float(arr.min())
    den = float(arr.max() - arr.min())
    if den < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return (arr / den * 255.0).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def save_debug_outputs(model: nn.Module, loader: DataLoader, device: torch.device, args) -> None:
    if not args.save_debug_vis or args.model_type != "lfaenet_tgfs_v2" or not hasattr(model, "set_debug_capture"):
        return
    out_dir = Path(args.save_dir) / "debug_vis"
    mask_dir = out_dir / "spatial_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    model.eval()
    model.set_debug_capture(True)
    saved = 0
    try:
        for batch in loader:
            if saved >= args.debug_vis_samples:
                break
            image, _, _, _ = forward_model(batch, model, args, device)
            debug = model.get_debug_outputs()
            names = batch.get("mask_name", [f"sample_{idx}" for idx in range(image.shape[0])])
            for item_idx in range(image.shape[0]):
                if saved >= args.debug_vis_samples:
                    break
                input_gray = to_uint8(image[item_idx].detach().cpu().squeeze(0).numpy())
                for stage_name in ("dec4", "dec3", "dec2", "dec1"):
                    stage_debug = debug.get(stage_name)
                    if stage_debug is None:
                        continue
                    spatial = stage_debug["spatial_mask"][item_idx : item_idx + 1]
                    spatial = F.interpolate(spatial, size=input_gray.shape, mode="bilinear", align_corners=False)
                    Image.fromarray(to_uint8(spatial.squeeze().numpy()), mode="L").save(
                        mask_dir / f"{saved:03d}_{stage_name}_{names[item_idx]}.png"
                    )
                    hh_scale = stage_debug.get("hh_scale")
                    rows.append(
                        {
                            "index": saved,
                            "mask_name": str(names[item_idx]),
                            "stage": stage_name,
                            "a_LL": float(stage_debug["a_ll_mean"][item_idx].item()),
                            "a_LH": float(stage_debug["a_lh_mean"][item_idx].item()),
                            "a_HL": float(stage_debug["a_hl_mean"][item_idx].item()),
                            "a_HH": float(stage_debug["a_hh_mean"][item_idx].item()),
                            "hh_scale": float(hh_scale[0].item()) if hh_scale is not None else float("nan"),
                            "lh_hl_scale": float(stage_debug["lh_hl_scale"][0].item()),
                        }
                    )
                saved += 1
    finally:
        model.set_debug_capture(False)

    if rows:
        csv_path = out_dir / "gate_stats.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["index", "mask_name", "stage", "a_LL", "a_LH", "a_HL", "a_HH", "hh_scale", "lh_hl_scale"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("Train FAENet / LFAENet-TGFS on QaTa-COV19-v2")
    parser.add_argument("--data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    parser.add_argument("--save-dir", type=str, default="runs/qata")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["faenet", "lfaenet_tgfs", "lfaenet_tgfs_v2"],
        default="lfaenet_tgfs_v2",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["poly", "cosine"],
        default="poly",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adamw"],
        default="sgd",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument(
        "--use-test-as-val",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        default=False,
    )

    parser.add_argument("--prompt-mode", type=str, choices=PROMPT_MODE_CHOICES, default="native")
    parser.add_argument("--use-cxr-bert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cxr-bert-dir", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--freeze-text-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument(
        "--drop-hh-in-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hh-drop-mode", type=str, choices=["zero", "keep", "learned"], default=None)
    parser.add_argument("--fusion-mode", type=str, choices=["encoder", "decoder", "both"], default="decoder")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--save-last-every", type=int, default=1)
    parser.add_argument("--save-best-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--abort-on-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-debug-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-vis-samples", type=int, default=8)
    parser.add_argument(
        "--use-deep-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--aux-w-d4", type=float, default=0.4)
    parser.add_argument("--aux-w-d3", type=float, default=0.6)
    parser.add_argument("--aux-w-d2", type=float, default=0.8)
    parser.add_argument("--low-level-hf-scale", type=float, default=0.6)
    parser.add_argument("--spatial-sharpen-power", type=float, default=2.0)

    args = parser.parse_args()
    if args.no_text and args.model_type != "faenet":
        raise ValueError("--no-text is only supported with --model-type faenet")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    args.save_last_every = max(1, int(args.save_last_every))
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.unfreeze_last_n = max(0, int(args.unfreeze_last_n))
    args.lora_r = max(0, int(args.lora_r))
    args.debug_vis_samples = max(1, int(args.debug_vis_samples))
    if args.hh_drop_mode is None:
        args.hh_drop_mode = "zero" if args.drop_hh_in_decoder else "keep"
    else:
        args.drop_hh_in_decoder = args.hh_drop_mode == "zero"
    if args.model_type != "lfaenet_tgfs_v2" and args.hh_drop_mode == "learned":
        raise ValueError("--hh-drop-mode learned is only implemented for --model-type lfaenet_tgfs_v2")
    if args.model_type != "lfaenet_tgfs_v2" and (args.unfreeze_last_n > 0 or args.lora_r > 0):
        raise ValueError("--unfreeze-last-n and --lora-r are only implemented for --model-type lfaenet_tgfs_v2")
    if not args.use_cxr_bert and (args.unfreeze_last_n > 0 or args.lora_r > 0):
        raise ValueError("--unfreeze-last-n/--lora-r require --use-cxr-bert")
    if args.spatial_sharpen_power <= 0:
        raise ValueError("--spatial-sharpen-power must be positive")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = bool(args.use_amp and device.type == "cuda" and args.model_type != "faenet")

    model, tokenizer = create_model(args, device)

    train_full_ds = QaTaCOV19Dataset(
        root_dir=args.data_root,
        split="train",
        image_size=args.image_size,
        use_text=not args.no_text,
        max_samples=args.max_train_samples,
    )
    test_ds = QaTaCOV19Dataset(
        root_dir=args.data_root,
        split="test",
        image_size=args.image_size,
        use_text=not args.no_text,
        max_samples=args.max_test_samples,
    )

    if args.use_test_as_val:
        train_ds = train_full_ds
        val_ds = test_ds
    else:
        total = len(train_full_ds)
        val_count = max(1, int(total * args.val_ratio))
        if val_count >= total:
            val_count = total - 1

        rng = random.Random(args.seed)
        indices = list(range(total))
        rng.shuffle(indices)
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        train_ds = Subset(train_full_ds, train_idx)
        val_ds = Subset(train_full_ds, val_idx)

    collate_fn = TextSegCollator(
        tokenizer=tokenizer if args.model_type != "faenet" else None,
        max_length=args.max_text_len,
        prompt_mode=args.prompt_mode,
        seed=args.seed,
        simple_vocab_size=args.vocab_size if args.model_type != "faenet" and tokenizer is None else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    criterion = SegLoss()
    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    thresholds = parse_thresholds(args.metric_thresholds)

    start_epoch = 1
    base_phase_lr = args.lr
    resumed_from: str | None = None
    best_threshold = 0.5
    if args.resume_ckpt is not None:
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            base_phase_lr = optimizer.param_groups[0]["lr"]

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        resumed_from = str(ckpt_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_log_path = save_dir / "epoch_log.txt"
    final_test_txt_path = save_dir / "final_test.txt"
    if args.resume_ckpt is None:
        txt_log_path.write_text("", encoding="utf-8")
        final_test_txt_path.write_text("", encoding="utf-8")

    config_path = save_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    append_log_line(txt_log_path, f"save_dir={save_dir}")
    append_log_line(txt_log_path, f"device={device}")
    append_log_line(
        txt_log_path,
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)}",
    )
    append_log_line(txt_log_path, json.dumps(vars(args), ensure_ascii=False))
    if resumed_from is not None:
        append_log_line(txt_log_path, f"resume_from={resumed_from} start_epoch={start_epoch}")

    best_dice = -1.0
    no_improve_epochs = 0
    history: list[dict[str, float]] = []
    history_path = save_dir / "history.json"
    if args.resume_ckpt is not None:
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if "best_dice" in ckpt:
            best_dice = float(ckpt["best_dice"])
        if "best_threshold" in ckpt:
            best_threshold = float(ckpt["best_threshold"])

    end_epoch = start_epoch + args.epochs - 1
    total_phase_epochs = max(args.epochs, 1)

    for epoch in range(start_epoch, end_epoch + 1):
        phase_epoch = epoch - start_epoch
        if args.lr_scheduler == "cosine":
            lr = cosine_lr(base_phase_lr, phase_epoch, total_phase_epochs, args.min_lr)
        else:
            lr = poly_lr(base_phase_lr, phase_epoch, total_phase_epochs, args.poly_power)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            args=args,
        )
        val_threshold_results, epoch_best_threshold = evaluate_thresholds(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            thresholds=thresholds,
        )
        val_stats = val_threshold_results[epoch_best_threshold]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_dice": val_stats["dice"],
            "val_threshold": epoch_best_threshold,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{end_epoch}] "
            f"lr={lr:.6f} "
            f"train: loss={row['train_loss']:.4f} iou={row['train_iou']:.4f} dice={row['train_dice']:.4f} | "
            f"val: loss={row['val_loss']:.4f} iou={row['val_iou']:.4f} dice={row['val_dice']:.4f} "
            f"thr={row['val_threshold']:.2f}"
        )
        append_log_line(
            txt_log_path,
            f"epoch={epoch:03d} lr={lr:.6f} "
            f"train_loss={row['train_loss']:.6f} train_iou={row['train_iou']:.6f} train_dice={row['train_dice']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_iou={row['val_iou']:.6f} val_dice={row['val_dice']:.6f} "
            f"val_thr={row['val_threshold']:.2f}",
        )

        if epoch == end_epoch or ((epoch - start_epoch) % args.save_last_every == 0):
            last_ckpt = save_dir / "last.pt"
            torch.save(
                build_checkpoint(model, optimizer, args, epoch, best_dice, best_threshold, include_optimizer=True),
                last_ckpt,
            )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_threshold = epoch_best_threshold
            no_improve_epochs = 0
            best_ckpt = save_dir / "best.pt"
            torch.save(
                build_checkpoint(
                    model,
                    optimizer,
                    args,
                    epoch,
                    best_dice,
                    best_threshold,
                    include_optimizer=args.save_best_optimizer,
                ),
                best_ckpt,
            )
        else:
            no_improve_epochs += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            append_log_line(
                txt_log_path,
                (
                    f"early_stop epoch={epoch} no_improve_epochs={no_improve_epochs} "
                    f"best_dice={best_dice:.6f} best_threshold={best_threshold:.2f}"
                ),
            )
            print(
                f"Early stopping at epoch {epoch}: "
                f"no improvement for {no_improve_epochs} epochs "
                f"(best_dice={best_dice:.4f}, best_threshold={best_threshold:.2f})"
            )
            break

    print(f"Training done. Best val dice: {best_dice:.4f} at threshold={best_threshold:.2f}")

    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        checkpoint = load_checkpoint(best_ckpt, device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        best_epoch = checkpoint.get("epoch", -1)
        best_threshold = float(checkpoint.get("best_threshold", best_threshold))
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        test_stats = validate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            args=args,
            threshold=best_threshold,
        )
        test_stats["best_epoch"] = int(best_epoch)
        test_stats["best_threshold"] = best_threshold
        test_summary = (
            f"best_epoch={best_epoch} best_threshold={best_threshold:.2f} "
            f"test_loss={test_stats['loss']:.6f} "
            f"test_iou={test_stats['iou']:.6f} "
            f"test_dice={test_stats['dice']:.6f}"
        )
        print(f"Final test with best checkpoint: {test_summary}")
        final_test_txt_path.write_text(test_summary + "\n", encoding="utf-8")
        append_log_line(txt_log_path, test_summary)
        (save_dir / "final_test.json").write_text(json.dumps(test_stats, indent=2), encoding="utf-8")
        save_debug_outputs(model, test_loader, device, args)


if __name__ == "__main__":
    main()
