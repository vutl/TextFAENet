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


def run_epoch(model, loader, criterion, device, optimizer=None, threshold: float = 0.5):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pred_pos_ratio = 0.0
    total_gt_pos_ratio = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(image)
            loss = criterion(logits, mask)

            if train_mode:
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


def evaluate_thresholds(model, loader, criterion, device, thresholds: list[float]) -> tuple[dict[float, dict[str, float]], float]:
    model.eval()
    results: dict[float, dict[str, float]] = {thr: {"loss": 0.0, "iou": 0.0, "dice": 0.0, "pred_pos_ratio": 0.0, "gt_pos_ratio": 0.0} for thr in thresholds}

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
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


def main() -> None:
    parser = argparse.ArgumentParser("Train FAENet on prepared MosMed 2D slices")
    parser.add_argument("--prepared-root", type=str, default="datasets/mosmed_2d_prepared")
    parser.add_argument("--save-dir", type=str, default="runs/mosmed_faenet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.2,0.3,0.4,0.5")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = FAENet(in_channels=1, num_classes=1).to(device)
    criterion = SegLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=resolved_pos_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    thresholds = parse_thresholds(args.metric_thresholds)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_path = save_dir / "epoch_log.txt"
    log_path.write_text("", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            "train_mask_stats "
            f"fg_fraction={train_mask_stats['fg_fraction']:.6f} "
            f"non_empty_fraction={train_mask_stats['non_empty_fraction']:.6f} "
            f"avg_area={train_mask_stats['avg_area']:.6f} "
            f"median_area={train_mask_stats['median_area']:.6f} "
            f"resolved_pos_weight={resolved_pos_weight:.4f}\n"
        )

    best_dice = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, threshold=0.5)
        val_threshold_results, epoch_best_threshold = evaluate_thresholds(model, val_loader, criterion, device, thresholds)
        val_stats = val_threshold_results[epoch_best_threshold]

        row = {
            "epoch": epoch,
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
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.6f} train_iou={row['train_iou']:.6f} train_dice={row['train_dice']:.6f} "
            f"train_pred_pos={row['train_pred_pos_ratio']:.6f} train_gt_pos={row['train_gt_pos_ratio']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_iou={row['val_iou']:.6f} val_dice={row['val_dice']:.6f} "
            f"val_pred_pos={row['val_pred_pos_ratio']:.6f} val_gt_pos={row['val_gt_pos_ratio']:.6f} "
            f"val_thr={row['val_threshold']:.2f}"
        )
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

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

        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best_ckpt = load_checkpoint(save_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))
    test_stats = run_epoch(model, test_loader, criterion, device, optimizer=None, threshold=best_threshold)

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

    print(f"Training complete. Final test: {summary}")


if __name__ == "__main__":
    main()
