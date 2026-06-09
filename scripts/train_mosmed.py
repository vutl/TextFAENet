from __future__ import annotations

import argparse
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
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import MosMed2DSegmentationDataset
from src.models import FAENet


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


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, float]:
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
    return {"iou": iou, "dice": dice, "pred_pos_ratio": pred_pos_ratio, "gt_pos_ratio": gt_pos_ratio}


def compute_foreground_stats(prepared_root: str, split: str, max_samples: int | None = None) -> dict[str, float]:
    root = Path(prepared_root)
    csv_path = root / "splits" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    import csv

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if max_samples is not None:
        rows = rows[:max_samples]

    fg_pixels = 0.0
    total_pixels = 0.0
    non_empty = 0
    areas: list[float] = []
    for row in rows:
        mask = np.asarray(Image.open(root / row["mask_path"]).convert("L"), dtype=np.float32) > 127
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
        frac = (epoch + 1) / max(warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * frac
    adj_epoch = epoch - warmup_epochs
    adj_total = max(max_epochs - warmup_epochs, 1)
    if scheduler == "cosine":
        return cosine_lr(base_lr, adj_epoch, adj_total, min_lr)
    return poly_lr(base_lr, adj_epoch, adj_total, poly_power)


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None, threshold: float = 0.5):
    from tqdm import tqdm

    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pred_pos_ratio = 0.0
    total_gt_pos_ratio = 0.0

    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    phase = "train" if train_mode else "val"
    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"  [{phase}]", leave=False, dynamic_ncols=True)

    for step, batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        use_amp = getattr(args, "use_amp", False)
        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, mask)

            if train_mode:
                scaled_loss = loss / accum_steps
                max_grad_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
                if use_amp and scaler is not None:
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

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.4f}", refresh=False)

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "pred_pos_ratio": total_pred_pos_ratio / n,
        "gt_pos_ratio": total_gt_pos_ratio / n,
    }


def evaluate_thresholds(model, loader, criterion, device, args, thresholds: list[float]) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    results: dict[float, dict[str, float]] = {thr: {"loss": 0.0, "iou": 0.0, "dice": 0.0, "pred_pos_ratio": 0.0, "gt_pos_ratio": 0.0} for thr in thresholds}

    use_amp = getattr(args, "use_amp", False)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, mask).item()
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


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def append_log_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Train FAENet (ResNet-50 encoder) on prepared MosMed 2D slices")
    parser.add_argument("--prepared-root", type=str, default="datasets/mosmed_2d_prepared")
    parser.add_argument("--save-dir", type=str, default="runs/mosmed_faenet_resnet50")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                        help="Separate LR for ResNet-50 encoder. Set 0 to use --lr for all params.")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, choices=["cosine", "poly"], default="cosine")
    parser.add_argument("--lr-warmup-epochs", type=int, default=10)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.2,0.3,0.4,0.5")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--encoder-type", type=str, choices=["from_scratch", "resnet50"], default="resnet50")
    parser.add_argument("--pretrained-image-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder-bn", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda"

    train_ds = MosMed2DSegmentationDataset(
        prepared_root=args.prepared_root,
        split="train",
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    val_ds = MosMed2DSegmentationDataset(
        prepared_root=args.prepared_root,
        split="val",
        image_size=args.image_size,
        max_samples=args.max_val_samples,
    )
    test_ds = MosMed2DSegmentationDataset(
        prepared_root=args.prepared_root,
        split="test",
        image_size=args.image_size,
        max_samples=args.max_test_samples,
    )

    train_mask_stats = compute_foreground_stats(args.prepared_root, "train", max_samples=args.max_train_samples)
    if args.pos_weight.lower() == "auto":
        resolved_pos_weight = auto_pos_weight_from_stats(train_mask_stats, args.max_pos_weight)
    else:
        resolved_pos_weight = float(args.pos_weight)
    args.resolved_pos_weight = resolved_pos_weight
    args.train_fg_fraction = train_mask_stats["fg_fraction"]
    args.train_non_empty_fraction = train_mask_stats["non_empty_fraction"]
    args.train_avg_mask_area = train_mask_stats["avg_area"]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = FAENet(
        in_channels=1,
        num_classes=1,
        encoder_type=args.encoder_type,
        pretrained_image_encoder=args.pretrained_image_encoder,
        freeze_encoder_bn=args.freeze_encoder_bn,
    ).to(device)

    criterion = SegLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=resolved_pos_weight,
    )

    encoder_lr = float(getattr(args, "encoder_lr", 0.0) or 0.0)
    use_separate_encoder_lr = (
        encoder_lr > 0
        and args.encoder_type == "resnet50"
        and hasattr(model, "image_encoder")
    )
    if use_separate_encoder_lr:
        encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        enc_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_param_ids]
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
    optimizer = AdamW(param_groups, eps=optim_eps)
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    thresholds = parse_thresholds(args.metric_thresholds)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_log_path = save_dir / "epoch_log.txt"
    history_path = save_dir / "history.json"

    start_epoch = 1
    best_dice = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []
    no_improve_epochs = 0

    if args.resume_ckpt is None:
        txt_log_path.write_text("", encoding="utf-8")
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

    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    append_log_line(txt_log_path, f"save_dir={save_dir}")
    append_log_line(txt_log_path, f"device={device}")
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

    end_epoch = start_epoch + args.epochs - 1
    total_phase_epochs = max(args.epochs, 1)
    warmup_epochs = max(0, int(args.lr_warmup_epochs))

    for epoch in range(start_epoch, end_epoch + 1):
        phase_epoch = epoch - start_epoch
        for pg in optimizer.param_groups:
            base_lr_g = float(pg.get("_base_lr", pg["lr"]))
            min_lr_g = float(pg.get("_min_lr", args.min_lr))
            pg["lr"] = lr_with_warmup(
                base_lr_g, phase_epoch, total_phase_epochs,
                warmup_epochs, args.lr_scheduler, min_lr_g, args.poly_power,
            )
        lr = optimizer.param_groups[-1]["lr"]

        train_stats = run_epoch(model, train_loader, criterion, device, args, optimizer=optimizer, scaler=scaler, threshold=0.5)
        val_threshold_results, epoch_best_threshold = evaluate_thresholds(model, val_loader, criterion, device, args, thresholds)
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
                "model_state": model.state_dict(),
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
                    "model_state": model.state_dict(),
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
                f"early_stop epoch={epoch} no_improve_epochs={no_improve_epochs} best_dice={best_dice:.6f}",
            )
            print(
                f"Early stopping at epoch {epoch}: no improvement for {no_improve_epochs} epochs "
                f"(best_dice={best_dice:.4f}, best_threshold={best_threshold:.2f})"
            )
            break

    best_ckpt = load_checkpoint(save_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))
    test_stats = run_epoch(model, test_loader, criterion, device, args, optimizer=None, threshold=best_threshold)

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
    with (save_dir / "final_test.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"best_epoch={summary['best_epoch']} best_threshold={summary['best_threshold']:.2f} "
            f"test_loss={summary['loss']:.6f} test_iou={summary['iou']:.6f} test_dice={summary['dice']:.6f} "
            f"test_pred_pos={summary['pred_pos_ratio']:.6f} test_gt_pos={summary['gt_pos_ratio']:.6f}\n"
        )
    append_log_line(txt_log_path, json.dumps(summary, ensure_ascii=False))
    print(f"Training complete. Final test: {summary}")


if __name__ == "__main__":
    main()
