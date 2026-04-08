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
from PIL import Image
from torch import nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import MosMed2DSegmentationDataset, MosMedTextCSVDataset, PromptedFolderSegmentationDataset
from src.models import FAENet, LFAENetTGFS, LFAENetTGFSv2


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
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
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
    pred_pos_ratio = preds.mean().item()
    gt_pos_ratio = targets.mean().item()
    return {
        "iou": iou,
        "dice": dice,
        "pred_pos_ratio": pred_pos_ratio,
        "gt_pos_ratio": gt_pos_ratio,
    }


class TextSegCollator:
    def __init__(self, tokenizer=None, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        masks = torch.stack([x["mask"] for x in batch], dim=0)
        texts = [str(x.get("text", "")) for x in batch]
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
            low_level_hf_scale=args.low_level_hf_scale,
            spatial_sharpen_power=args.spatial_sharpen_power,
            use_deep_supervision=args.use_deep_supervision,
        )

    if args.model_type != "faenet" and args.use_cxr_bert:
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


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None, threshold: float = 0.5):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pred_pos_ratio = 0.0
    total_gt_pos_ratio = 0.0

    for batch in loader:
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                _, mask, logits, aux = forward_model(batch, model, args, device)
                loss = compute_loss_with_aux(
                    criterion,
                    logits,
                    mask,
                    aux,
                    args.aux_w_d4,
                    args.aux_w_d3,
                    args.aux_w_d2,
                )

            if train_mode:
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        m = batch_metrics(logits.detach(), mask, threshold=threshold)
        total_loss += loss.item()
        total_iou += m["iou"]
        total_dice += m["dice"]
        total_pred_pos_ratio += m["pred_pos_ratio"]
        total_gt_pos_ratio += m["gt_pos_ratio"]

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "pred_pos_ratio": total_pred_pos_ratio / n,
        "gt_pos_ratio": total_gt_pos_ratio / n,
    }


@torch.no_grad()
def evaluate_thresholds(model, loader, criterion, device, args, thresholds: list[float]) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    results: dict[float, dict[str, float]] = {
        thr: {"loss": 0.0, "iou": 0.0, "dice": 0.0, "pred_pos_ratio": 0.0, "gt_pos_ratio": 0.0}
        for thr in thresholds
    }

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
            _, mask, logits, aux = forward_model(batch, model, args, device)
            loss = compute_loss_with_aux(
                criterion,
                logits,
                mask,
                aux,
                args.aux_w_d4,
                args.aux_w_d3,
                args.aux_w_d2,
            ).item()

        for thr in thresholds:
            m = batch_metrics(logits, mask, threshold=thr)
            results[thr]["loss"] += loss
            results[thr]["iou"] += m["iou"]
            results[thr]["dice"] += m["dice"]
            results[thr]["pred_pos_ratio"] += m["pred_pos_ratio"]
            results[thr]["gt_pos_ratio"] += m["gt_pos_ratio"]

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
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_dataset(args, split: str):
    max_samples = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }[split]
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
        )
    return PromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split=split,
        image_size=args.image_size,
        max_samples=max_samples,
    )


def checkpoint_state_dict(model: nn.Module, args) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if args.model_type in {"lfaenet_tgfs", "lfaenet_tgfs_v2"} and args.use_cxr_bert and args.freeze_text_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("text_encoder.model.")}
    return state


def main() -> None:
    parser = argparse.ArgumentParser("Train LFAENet-TGFS on MosMed")
    parser.add_argument("--data-root", type=str, default="datasets/MosMed")
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["prepared", "text_csv", "prompt_folder"],
        default="text_csv",
    )
    parser.add_argument("--save-dir", type=str, default="runs/mosmed_text_faenet")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["faenet", "lfaenet_tgfs", "lfaenet_tgfs_v2"],
        default="lfaenet_tgfs_v2",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
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
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.2,0.3,0.4,0.5")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--no-text", action="store_true", default=False)
    parser.add_argument("--early-stop-patience", type=int, default=8)

    parser.add_argument("--use-cxr-bert", action="store_true", default=True)
    parser.add_argument("--cxr-bert-dir", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--freeze-text-backbone", action="store_true", default=True)
    parser.add_argument("--drop-hh-in-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--use-deep-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-w-d4", type=float, default=0.4)
    parser.add_argument("--aux-w-d3", type=float, default=0.6)
    parser.add_argument("--aux-w-d2", type=float, default=0.8)
    parser.add_argument("--low-level-hf-scale", type=float, default=0.6)
    parser.add_argument("--spatial-sharpen-power", type=float, default=2.0)

    args = parser.parse_args()
    if args.no_text and args.model_type != "faenet":
        raise ValueError("--no-text is only supported with --model-type faenet")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = device.type == "cuda" and args.model_type != "faenet"

    train_ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val")
    test_ds = build_dataset(args, "test")

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
        tokenizer=tokenizer if args.model_type != "faenet" else None,
        max_length=args.max_text_len,
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
    )
    if args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    append_log_line(txt_log_path, f"save_dir={save_dir}")
    append_log_line(txt_log_path, f"device={device}")
    append_log_line(
        txt_log_path,
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)}",
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

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_threshold = epoch_best_threshold
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model, args),
                    "optimizer_state": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_threshold": best_threshold,
                    "args": vars(args),
                },
                save_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

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
        "loss": float(test_stats["loss"]),
        "iou": float(test_stats["iou"]),
        "dice": float(test_stats["dice"]),
        "pred_pos_ratio": float(test_stats["pred_pos_ratio"]),
        "gt_pos_ratio": float(test_stats["gt_pos_ratio"]),
    }
    (save_dir / "final_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_test_txt_path.write_text(
        (
            f"best_epoch={summary['best_epoch']} best_threshold={summary['best_threshold']:.2f} "
            f"test_loss={summary['loss']:.6f} test_iou={summary['iou']:.6f} test_dice={summary['dice']:.6f} "
            f"test_pred_pos={summary['pred_pos_ratio']:.6f} test_gt_pos={summary['gt_pos_ratio']:.6f}\n"
        ),
        encoding="utf-8",
    )
    append_log_line(txt_log_path, json.dumps(summary, ensure_ascii=False))
    print(f"Training complete. Final test: {summary}")


if __name__ == "__main__":
    main()
