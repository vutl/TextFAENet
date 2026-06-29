from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FMISEG_ROOT = ROOT.parent / "FMISeg"
LEGACY_COMMIT = "c22b6d7"


def write_legacy_module(fmiseg_root: Path, cache_root: Path) -> Path:
    net_dir = cache_root / "net"
    net_dir.mkdir(parents=True, exist_ok=True)
    (net_dir / "__init__.py").write_text("", encoding="utf-8")

    for rel_path in ("net/model.py", "net/decoder.py"):
        source = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={fmiseg_root.as_posix()}",
                "-C",
                str(fmiseg_root),
                "show",
                f"{LEGACY_COMMIT}:{rel_path}",
            ],
            text=True,
        )
        (cache_root / rel_path).write_text(source, encoding="utf-8")

    return cache_root


def load_cfg(fmiseg_root: Path, config_path: Path) -> SimpleNamespace:
    sys.path.insert(0, str(fmiseg_root))
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


def dice_iou_from_probs(probs: torch.Tensor, target: torch.Tensor, threshold: float):
    pred = (probs > threshold).float()
    target = target.float()
    flat_pred = pred.flatten(1)
    flat_target = target.flatten(1)
    intersection = (flat_pred * flat_target).sum(dim=1)
    pred_sum = flat_pred.sum(dim=1)
    target_sum = flat_target.sum(dim=1)
    union = ((flat_pred + flat_target) > 0).float().sum(dim=1)
    eps = 1e-6
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou, intersection, union, pred_sum, target_sum


def summarize(rows: list[dict], prefix: str) -> dict:
    dice = np.asarray([row[f"{prefix}_dice"] for row in rows], dtype=np.float64)
    iou = np.asarray([row[f"{prefix}_iou"] for row in rows], dtype=np.float64)
    inter = float(sum(row[f"{prefix}_intersection"] for row in rows))
    union = float(sum(row[f"{prefix}_union"] for row in rows))
    pred = float(sum(row[f"{prefix}_pred_pixels"] for row in rows))
    target = float(sum(row["target_pixels"] for row in rows))
    eps = 1e-6
    return {
        f"{prefix}_per_image_dice": float(dice.mean()),
        f"{prefix}_per_image_iou": float(iou.mean()),
        f"{prefix}_global_dice": float((2.0 * inter + eps) / (pred + target + eps)),
        f"{prefix}_global_iou": float((inter + eps) / (union + eps)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Evaluate legacy FMISeg QaTa checkpoint with per-image and global metrics.")
    parser.add_argument("--fmiseg-root", type=Path, default=DEFAULT_FMISEG_ROOT)
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--checkpoint", type=str, default="save_model/last.ckpt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=ROOT / "fmiseg_qata_legacy_dual_metrics_20260628.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "fmiseg_qata_legacy_dual_metrics_20260628.md")
    args = parser.parse_args()

    fmiseg_root = args.fmiseg_root.resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = fmiseg_root / config_path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = fmiseg_root / checkpoint_path

    cache_root = write_legacy_module(fmiseg_root, ROOT / ".fmiseg_legacy_eval")
    sys.path.insert(0, str(cache_root))
    sys.path.insert(1, str(fmiseg_root))

    cfg = load_cfg(fmiseg_root, config_path)
    cfg.bert_type = resolve_path(fmiseg_root, cfg.bert_type)
    cfg.vision_type = resolve_path(fmiseg_root, cfg.vision_type)
    cfg.test_csv_path = resolve_path(fmiseg_root, cfg.test_csv_path)
    cfg.test_root_path = resolve_path(fmiseg_root, cfg.test_root_path)
    cfg.image_size = list(cfg.image_size)

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

    from net.model import SegModel
    from utils.dataset import SegData

    device = torch.device(args.device)
    model = SegModel(cfg.bert_type, cfg.vision_type, cfg.project_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    model_state = {}
    for key, value in checkpoint.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value

    incompatible = model.load_state_dict(model_state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if key != "text_encoder.model.bert.embeddings.position_ids"
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Legacy FMISeg model mismatch: "
            f"missing={incompatible.missing_keys[:10]} unexpected={unexpected[:10]}"
        )

    dataset = SegData(
        dataname=getattr(cfg, "dataset_name", "cov19"),
        csv_path=cfg.test_csv_path,
        root_path=cfg.test_root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode="test",
        wavelet_type=getattr(cfg, "wavelet_type", "haar"),
        auto_prompt_from_mask=getattr(cfg, "auto_prompt_from_mask", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    rows: list[dict] = []
    for batch_idx, (data, target) in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        data, target = move_batch_to_device(data, target, device)
        probs_first, probs_second = model(data)
        probs_ensemble = (probs_first + probs_second) / 2.0
        for prefix, probs in (("first_branch", probs_first), ("ensemble", probs_ensemble)):
            dice, iou, inter, union, pred_sum, target_sum = dice_iou_from_probs(
                probs.float(),
                target.float(),
                args.threshold,
            )
            if prefix == "first_branch":
                for index in range(target.shape[0]):
                    rows.append({
                        "batch_index": batch_idx,
                        "sample_index": batch_idx * args.batch_size + index,
                        "target_pixels": float(target_sum[index].item()),
                    })
            offset = len(rows) - target.shape[0]
            for index in range(target.shape[0]):
                row = rows[offset + index]
                row[f"{prefix}_dice"] = float(dice[index].item())
                row[f"{prefix}_iou"] = float(iou[index].item())
                row[f"{prefix}_intersection"] = float(inter[index].item())
                row[f"{prefix}_union"] = float(union[index].item())
                row[f"{prefix}_pred_pixels"] = float(pred_sum[index].item())

    result = {
        "dataset": "QaTa-COV19-v2",
        "model": "FMISeg legacy checkpoint",
        "fmiseg_root": str(fmiseg_root),
        "legacy_commit": LEGACY_COMMIT,
        "checkpoint": str(checkpoint_path),
        "threshold": args.threshold,
        "num_images": len(rows),
        **summarize(rows, "first_branch"),
        **summarize(rows, "ensemble"),
        "note": "Original FMISeg wrapper evaluated the first branch only; ensemble averages the two branch probabilities.",
    }

    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        "# FMISeg QaTa Legacy Dual Metrics",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Legacy model files: git commit `{LEGACY_COMMIT}`",
        f"- Threshold: `{args.threshold}`",
        f"- Test images: `{len(rows)}`",
        "",
        "| Output | Per-image Dice | Global Dice | Per-image IoU | Global IoU |",
        "|---|---:|---:|---:|---:|",
        (
            f"| First branch (original wrapper) | {result['first_branch_per_image_dice']:.6f} | "
            f"{result['first_branch_global_dice']:.6f} | {result['first_branch_per_image_iou']:.6f} | "
            f"{result['first_branch_global_iou']:.6f} |"
        ),
        (
            f"| Two-branch ensemble | {result['ensemble_per_image_dice']:.6f} | "
            f"{result['ensemble_global_dice']:.6f} | {result['ensemble_per_image_iou']:.6f} | "
            f"{result['ensemble_global_iou']:.6f} |"
        ),
        "",
        "Original FMISeg code returned only the first output branch to TorchMetrics. The ensemble row is added for reference.",
    ]
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
