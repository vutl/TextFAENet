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
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import QaTaCOV19Dataset
from src.models import FAENet, LFAENetTGFS, LFAENetTGFSv2


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


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    tgt_sum = targets.sum(dim=(1, 2, 3))

    iou = ((inter + eps) / (union + eps)).mean().item()
    dice = ((2 * inter + eps) / (pred_sum + tgt_sum + eps)).mean().item()
    return {"iou": iou, "dice": dice}


class TextSegCollator:
    def __init__(self, tokenizer=None, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = torch.stack([x["image"] for x in batch], dim=0)
        masks = torch.stack([x["mask"] for x in batch], dim=0)
        texts = [x["text"] for x in batch]
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, args):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
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
            loss = compute_loss_with_aux(
                criterion,
                logits,
                mask,
                aux,
                args.aux_w_d4,
                args.aux_w_d3,
                args.aux_w_d2,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for batch in loader:
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

        loss = compute_loss_with_aux(
            criterion,
            logits,
            mask,
            aux,
            args.aux_w_d4,
            args.aux_w_d3,
            args.aux_w_d2,
        )
        m = batch_metrics(logits, mask)

        total_loss += loss.item()
        total_iou += m["iou"]
        total_dice += m["dice"]

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
    }


def poly_lr(base_lr: float, epoch: int, max_epochs: int, power: float) -> float:
    return base_lr * ((1.0 - (epoch / max_epochs)) ** power)


def append_log_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def checkpoint_state_dict(model: nn.Module, args) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if args.model_type in {"lfaenet_tgfs", "lfaenet_tgfs_v2"} and args.use_cxr_bert and args.freeze_text_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("text_encoder.model.")}
    return state


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
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--use-cxr-bert", action="store_true", default=True)
    parser.add_argument("--cxr-bert-dir", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--freeze-text-backbone", action="store_true", default=True)
    parser.add_argument(
        "--drop-hh-in-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--resume-ckpt", type=str, default=None)
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
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = create_model(args, device)

    train_ds = QaTaCOV19Dataset(
        root_dir=args.data_root,
        split="train",
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    test_ds = QaTaCOV19Dataset(
        root_dir=args.data_root,
        split="test",
        image_size=args.image_size,
        max_samples=args.max_test_samples,
    )

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    criterion = SegLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    base_phase_lr = args.lr
    resumed_from: str | None = None
    if args.resume_ckpt is not None:
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
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
    append_log_line(txt_log_path, f"train_samples={len(train_ds)} test_samples={len(test_ds)}")
    append_log_line(txt_log_path, json.dumps(vars(args), ensure_ascii=False))
    if resumed_from is not None:
        append_log_line(txt_log_path, f"resume_from={resumed_from} start_epoch={start_epoch}")

    best_dice = -1.0
    history: list[dict[str, float]] = []
    history_path = save_dir / "history.json"
    if args.resume_ckpt is not None and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if "best_dice" in ckpt:
            best_dice = float(ckpt["best_dice"])

    end_epoch = start_epoch + args.epochs - 1
    total_phase_epochs = max(args.epochs, 1)

    for epoch in range(start_epoch, end_epoch + 1):
        phase_epoch = epoch - start_epoch
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
        val_stats = validate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_dice": val_stats["dice"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{end_epoch}] "
            f"lr={lr:.6f} "
            f"train: loss={row['train_loss']:.4f} iou={row['train_iou']:.4f} dice={row['train_dice']:.4f} | "
            f"val: loss={row['val_loss']:.4f} iou={row['val_iou']:.4f} dice={row['val_dice']:.4f}"
        )
        append_log_line(
            txt_log_path,
            f"epoch={epoch:03d} lr={lr:.6f} "
            f"train_loss={row['train_loss']:.6f} train_iou={row['train_iou']:.6f} train_dice={row['train_dice']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_iou={row['val_iou']:.6f} val_dice={row['val_dice']:.6f}",
        )

        last_ckpt = save_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": checkpoint_state_dict(model, args),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_dice": best_dice,
            },
            last_ckpt,
        )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_ckpt = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model, args),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "best_dice": best_dice,
                },
                best_ckpt,
            )

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training done. Best val dice: {best_dice:.4f}")

    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        best_epoch = checkpoint.get("epoch", -1)
        test_stats = validate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            args=args,
        )
        test_summary = (
            f"best_epoch={best_epoch} "
            f"test_loss={test_stats['loss']:.6f} "
            f"test_iou={test_stats['iou']:.6f} "
            f"test_dice={test_stats['dice']:.6f}"
        )
        print(f"Final test with best checkpoint: {test_summary}")
        final_test_txt_path.write_text(test_summary + "\n", encoding="utf-8")
        append_log_line(txt_log_path, test_summary)
        (save_dir / "final_test.json").write_text(json.dumps(test_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
