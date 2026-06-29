from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FMISEG_ROOT = ROOT.parent / "FMISeg"


def load_cfg(code_root: Path, config_path: Path):
    sys.path.insert(0, str(code_root))
    from utils import config as fmiseg_config

    return fmiseg_config.load_cfg_from_cfg_file(str(config_path))


def resolve_path(base: Path, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def move_batch_to_device(data, target: torch.Tensor, device: torch.device):
    high, low, text = data
    text = {key: value.to(device) for key, value in text.items()}
    return [high.to(device), low.to(device), text], target.to(device)


def per_image_counts(probs: torch.Tensor, target: torch.Tensor, threshold: float):
    preds = (probs > threshold).float()
    target = target.float()
    flat_pred = preds.flatten(1)
    flat_target = target.flatten(1)
    inter = (flat_pred * flat_target).sum(dim=1)
    pred_sum = flat_pred.sum(dim=1)
    target_sum = flat_target.sum(dim=1)
    union = ((flat_pred + flat_target) > 0).float().sum(dim=1)
    eps = 1e-6
    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou, inter, union, pred_sum, target_sum


def save_binary_masks(
    probs: torch.Tensor,
    names: list[str],
    pred_dir: Path,
    threshold: float,
    transpose: bool = False,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds = (probs.detach().cpu().numpy()[:, 0] > threshold).astype(np.uint8) * 255
    for pred, name in zip(preds, names):
        if transpose:
            pred = pred.T
        Image.fromarray(pred, mode="L").save(pred_dir / name)


def get_mask_name(dataset, index: int) -> str:
    if hasattr(dataset, "records"):
        return Path(dataset.records[index]["mask_path"]).name
    if hasattr(dataset, "image_list"):
        return Path(dataset.image_list[index]).name
    raise AttributeError("Cannot infer mask filename from FMISeg dataset object.")


def build_segdata(SegData, cfg):
    common = dict(
        dataname=getattr(cfg, "dataset_name", "cov19"),
        csv_path=cfg.test_csv_path,
        root_path=cfg.test_root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode="test",
    )
    try:
        return SegData(
            **common,
            wavelet_type=getattr(cfg, "wavelet_type", "haar"),
            auto_prompt_from_mask=getattr(cfg, "auto_prompt_from_mask", False),
        )
    except TypeError:
        return SegData(**common)


def summarize(rows: list[dict]) -> dict:
    dice = np.asarray([row["dice"] for row in rows], dtype=np.float64)
    iou = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    inter = float(sum(row["intersection"] for row in rows))
    union = float(sum(row["union"] for row in rows))
    pred = float(sum(row["pred_pixels"] for row in rows))
    target = float(sum(row["target_pixels"] for row in rows))
    eps = 1e-6
    return {
        "num_images": len(rows),
        "per_image_dice": float(dice.mean()),
        "per_image_iou": float(iou.mean()),
        "global_dice": float((2.0 * inter + eps) / (pred + target + eps)),
        "global_iou": float((inter + eps) / (union + eps)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Export valid FMISeg official QaTa per-image metrics.")
    parser.add_argument("--fmiseg-root", type=Path, default=DEFAULT_FMISEG_ROOT)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="Optional FMISeg code root. Use this for historical checkpoint-compatible code snapshots.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Optional root for relative data/lib/checkpoint paths. Defaults to --fmiseg-root.",
    )
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--outputs-are-probs",
        action="store_true",
        help="Set for legacy FMISeg checkpoints whose forward pass already returns sigmoid probabilities.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "external_metrics" / "fmiseg_qata_official")
    parser.add_argument(
        "--save-pred-dir",
        type=Path,
        default=None,
        help="Optional directory for thresholded prediction masks. Defaults to <output-dir>/pred_masks.",
    )
    parser.add_argument(
        "--transpose-saved-pred-masks",
        action="store_true",
        help="Transpose saved masks for Text-FAENet/PIL QaTa visualization orientation.",
    )
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if args.save_pred_dir is not None and not args.save_pred_dir.is_absolute():
        args.save_pred_dir = ROOT / args.save_pred_dir

    fmiseg_root = args.fmiseg_root.resolve()
    code_root = (args.code_root or fmiseg_root).resolve()
    asset_root = (args.asset_root or fmiseg_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = asset_root / config_path
    cfg = load_cfg(code_root, config_path)
    cfg.bert_type = resolve_path(asset_root, cfg.bert_type)
    cfg.vision_type = resolve_path(asset_root, cfg.vision_type)
    cfg.test_csv_path = resolve_path(asset_root, cfg.test_csv_path)
    cfg.test_root_path = resolve_path(asset_root, cfg.test_root_path)
    cfg.image_size = list(cfg.image_size)

    checkpoint_path = Path(args.checkpoint or getattr(cfg, "checkpoint_path", "") or "")
    if not checkpoint_path:
        model_dir = Path(resolve_path(asset_root, getattr(cfg, "model_save_path", "save_model")))
        best_txt = model_dir / "best_checkpoint.txt"
        checkpoint_path = model_dir / "last-v1.ckpt"
        if best_txt.exists():
            for line in best_txt.read_text(encoding="utf-8").splitlines():
                if line.startswith("best_model_path="):
                    candidate = Path(line.split("=", 1)[1].strip())
                    if candidate.exists():
                        checkpoint_path = candidate
                        break
        if not checkpoint_path.exists() and (model_dir / "last.ckpt").exists():
            checkpoint_path = model_dir / "last.ckpt"
    elif not checkpoint_path.is_absolute():
        checkpoint_path = asset_root / checkpoint_path

    sys.path.insert(0, str(code_root))
    from net.creratemodel import CreateModel
    from utils.dataset import SegData

    device = torch.device(args.device)
    model = CreateModel(cfg).to(device)
    try:
        ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    ckpt = ckpt_obj["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    dataset = build_segdata(SegData, cfg)
    batch_size = int(args.batch_size or getattr(cfg, "valid_batch_size", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict] = []
    pred_dir = args.save_pred_dir or (args.output_dir / "pred_masks")
    for batch_idx, (data, target) in enumerate(loader):
        data, target = move_batch_to_device(data, target, device)
        logits1, logits2 = model(data)
        if args.outputs_are_probs:
            probs = (logits1.float() + logits2.float()) / 2.0
        else:
            probs = (torch.sigmoid(logits1.float()) + torch.sigmoid(logits2.float())) / 2.0
        dice, iou, inter, union, pred_sum, target_sum = per_image_counts(probs, target.float(), args.threshold)
        start = batch_idx * batch_size
        names = [get_mask_name(dataset, start + index) for index in range(target.shape[0])]
        save_binary_masks(probs, names, pred_dir, args.threshold, transpose=args.transpose_saved_pred_masks)
        for index in range(target.shape[0]):
            name = names[index]
            rows.append(
                {
                    "mask_name": name,
                    "dice": float(dice[index].detach().cpu()),
                    "iou": float(iou[index].detach().cpu()),
                    "intersection": float(inter[index].detach().cpu()),
                    "union": float(union[index].detach().cpu()),
                    "pred_pixels": float(pred_sum[index].detach().cpu()),
                    "target_pixels": float(target_sum[index].detach().cpu()),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "test_per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mask_name", "dice", "iou", "intersection", "union", "pred_pixels", "target_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "QaTa-COV19-v2",
        "model": "FMISeg official",
        "checkpoint": str(checkpoint_path),
        "code_root": str(code_root),
        "asset_root": str(asset_root),
        "threshold": args.threshold,
        "per_image_csv": str(csv_path.relative_to(ROOT)),
        "pred_mask_dir": str(pred_dir.resolve()),
        **summarize(rows),
        "note": "Computed with FMISeg net.creratemodel.CreateModel official ensemble logits, matching evaluate.py semantics.",
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = args.output_dir / "summary.md"
    md_path.write_text(
        "\n".join(
            [
                "# FMISeg Official QaTa Metrics",
                "",
                f"- Checkpoint: `{summary['checkpoint']}`",
                f"- Per-image CSV: `{summary['per_image_csv']}`",
                f"- Threshold: `{summary['threshold']}`",
                "",
                "| Per-image Dice | Global Dice | Per-image IoU | Global IoU |",
                "|---:|---:|---:|---:|",
                (
                    f"| {summary['per_image_dice']:.6f} | {summary['global_dice']:.6f} | "
                    f"{summary['per_image_iou']:.6f} | {summary['global_iou']:.6f} |"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
