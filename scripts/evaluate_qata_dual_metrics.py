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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

from src.data import QaTaCOV19Dataset
from scripts.train_qata import (
    TEXT_MODEL_TYPES,
    VISUAL_ONLY_MODEL_TYPES,
    TextSegCollator,
    create_model,
    forward_model,
    load_checkpoint,
    set_seed,
)


CONFIG_DEFAULTS = {
    "abort_on_nonfinite": True,
    "cxr_bert_dir": "BiomedVLP-CXR-BERT-specialized",
    "drop_hh_in_decoder": False,
    "dropout_p": 0.0,
    "encoder_text_fusion": "film",
    "freeze_encoder_bn": True,
    "freeze_text_backbone": True,
    "freq_drop_bands": "none",
    "fusion_mode": "decoder",
    "grounding_n_heads": 1,
    "hh_drop_mode": "keep",
    "learnable_low_level_hf_scale": False,
    "learnable_spatial_sharpen": False,
    "lora_r": 0,
    "low_level_hf_scale": 0.6,
    "max_text_len": 64,
    "model_type": "lfaenet_tgfs_v2",
    "norm_type": "bn",
    "pretrained_image_encoder": True,
    "prompt_mode": "native",
    "seed": 42,
    "spatial_sharpen_power": 2.0,
    "text_dim": 256,
    "text_pooling": "mean",
    "unfreeze_last_n": 0,
    "use_cxr_bert": True,
    "use_deep_supervision": False,
    "v3_encoder_type": "from_scratch",
    "visual_pretrained": "none",
    "vocab_size": 30522,
}


LEGACY_WAVELET_BUFFER_SUFFIXES = (
    ".dwt.dwt.h0_col",
    ".dwt.dwt.h1_col",
    ".dwt.dwt.h0_row",
    ".dwt.dwt.h1_row",
    ".idwt.idwt.g0_col",
    ".idwt.idwt.g1_col",
    ".idwt.idwt.g0_row",
    ".idwt.idwt.g1_row",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def namespace_from_config(config: dict, use_amp: bool) -> argparse.Namespace:
    resolved = dict(CONFIG_DEFAULTS)
    resolved.update(config)
    resolved["use_amp"] = bool(use_amp and torch.cuda.is_available())
    return argparse.Namespace(**resolved)


def checkpoint_path(run_dir: Path) -> Path:
    best = run_dir / "best.pt"
    if best.exists():
        return best
    last = run_dir / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No best.pt or last.pt found in {run_dir}")


def _same_tensor_shape(a: object, b: object) -> bool:
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        return True
    return tuple(a.shape) == tuple(b.shape)


def _legacy_stage_key(key: str) -> str:
    # Older LFAENet-TGFS checkpoints stored encoder stages as Sequential:
    #   enc1.0.* -> ConvBlock, enc1.1.* -> FreqA/TGFS
    # Current checkpoints name the same modules explicitly:
    #   enc1.block.*, enc1.freqa.*
    for stage in ("enc1", "enc2", "enc3", "enc4", "bottleneck"):
        old_block = f"{stage}.0."
        if key.startswith(old_block):
            return f"{stage}.block.{key[len(old_block):]}"
        old_freqa = f"{stage}.1."
        if key.startswith(old_freqa):
            return f"{stage}.freqa.{key[len(old_freqa):]}"
    return key


def normalize_checkpoint_state_for_current_model(
    state: dict,
    model_state: dict,
) -> tuple[dict, dict[str, int]]:
    normalized = {}
    stats = {"remapped": 0, "dropped_legacy_wavelet_buffers": 0}
    for key, value in state.items():
        target_key = key if key in model_state else _legacy_stage_key(key)
        if (
            target_key != key
            and target_key in model_state
            and _same_tensor_shape(value, model_state[target_key])
        ):
            normalized[target_key] = value
            stats["remapped"] += 1
            continue

        if key not in model_state and key.endswith(LEGACY_WAVELET_BUFFER_SUFFIXES):
            stats["dropped_legacy_wavelet_buffers"] += 1
            continue

        normalized[key] = value
    return normalized, stats


@torch.inference_mode()
def evaluate_run(
    run_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> dict:
    config = read_json(run_dir / "config.json")
    args = namespace_from_config(config, use_amp=use_amp)
    set_seed(int(args.seed))

    model, tokenizer = create_model(args, device)
    ckpt_path = checkpoint_path(run_dir)
    checkpoint = load_checkpoint(ckpt_path, device)
    state = checkpoint.get("model_state", checkpoint)
    state, load_stats = normalize_checkpoint_state_for_current_model(
        state,
        model.state_dict(),
    )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("text_encoder.model.")
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch for {run_dir.name}: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.eval()

    final_path = run_dir / "final_test.json"
    final = read_json(final_path) if final_path.exists() else {}
    threshold = float(final.get("best_threshold", checkpoint.get("best_threshold", 0.5)))
    use_text = args.model_type not in VISUAL_ONLY_MODEL_TYPES
    dataset = QaTaCOV19Dataset(
        root_dir=args.data_root,
        split="test",
        image_size=int(args.image_size),
        use_text=use_text,
        max_samples=None,
    )
    collator = TextSegCollator(
        tokenizer=tokenizer if args.model_type in TEXT_MODEL_TYPES else None,
        max_length=int(args.max_text_len),
        prompt_mode=str(args.prompt_mode),
        seed=int(args.seed),
        simple_vocab_size=(
            int(args.vocab_size)
            if args.model_type in TEXT_MODEL_TYPES and tokenizer is None
            else None
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collator,
    )

    rows: list[dict[str, float | str]] = []
    total_intersection = 0.0
    total_union = 0.0
    total_pred = 0.0
    total_target = 0.0
    eps = 1e-6

    for batch in loader:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=args.use_amp,
        ):
            _, targets, logits, _ = forward_model(batch, model, args, device)

        predictions = (torch.sigmoid(logits.float()) > threshold).float()
        targets = targets.float()
        intersection = (predictions * targets).sum(dim=(1, 2, 3))
        union = ((predictions + targets) > 0).float().sum(dim=(1, 2, 3))
        pred_sum = predictions.sum(dim=(1, 2, 3))
        target_sum = targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        iou = (intersection + eps) / (union + eps)

        for index, name in enumerate(batch["mask_name"]):
            rows.append(
                {
                    "mask_name": str(name),
                    "dice": float(dice[index].item()),
                    "iou": float(iou[index].item()),
                    "intersection": float(intersection[index].item()),
                    "pred_pixels": float(pred_sum[index].item()),
                    "target_pixels": float(target_sum[index].item()),
                }
            )

        total_intersection += float(intersection.sum().item())
        total_union += float(union.sum().item())
        total_pred += float(pred_sum.sum().item())
        total_target += float(target_sum.sum().item())

    dice_values = np.asarray([float(row["dice"]) for row in rows], dtype=np.float64)
    iou_values = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    per_image_csv = run_dir / "test_per_image_metrics.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "run": run_dir.name,
        "checkpoint": ckpt_path.name,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold": threshold,
        "num_images": len(rows),
        "per_image_dice": float(dice_values.mean()),
        "per_image_dice_std": float(dice_values.std()),
        "per_image_dice_median": float(np.median(dice_values)),
        "per_image_iou": float(iou_values.mean()),
        "per_image_iou_std": float(iou_values.std()),
        "per_image_iou_median": float(np.median(iou_values)),
        "global_dice": float(
            (2.0 * total_intersection + eps) / (total_pred + total_target + eps)
        ),
        "global_iou": float((total_intersection + eps) / (total_union + eps)),
        "old_final_dice": final.get("dice"),
        "old_final_iou": final.get("iou"),
        "inference_amp": bool(args.use_amp),
        "per_image_csv": str(per_image_csv.relative_to(ROOT)),
        "model_type": args.model_type,
        "visual_encoder": (
            args.visual_pretrained
            if args.model_type.startswith("resnet50_")
            else args.v3_encoder_type
            if args.model_type == "lfaenet_tgfs_v3"
            else "from_scratch"
        ),
        "text_encoder": "cxr_bert" if args.use_cxr_bert else "simple",
        "fusion_mode": args.fusion_mode,
        "hh_drop_mode": args.hh_drop_mode,
        "legacy_state_remapped": int(load_stats["remapped"]),
        "legacy_wavelet_buffers_dropped": int(load_stats["dropped_legacy_wavelet_buffers"]),
    }
    (run_dir / "test_dual_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# QaTa Dual-Metric Evaluation",
        "",
        "`Per-image` is the arithmetic mean of image-level scores. `Global` pools all test pixels before computing the score.",
        "",
        "| Run | Checkpoint | Thr | Images | Per-image Dice | Global Dice | Per-image IoU | Global IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| `{row['run']}` | epoch {row['checkpoint_epoch']} | {row['threshold']:.2f} | "
            f"{row['num_images']} | {row['per_image_dice']:.6f} | {row['global_dice']:.6f} | "
            f"{row['per_image_iou']:.6f} | {row['global_iou']:.6f} |"
        )
    lines.extend(["", "## Configuration", ""])
    for row in results:
        lines.append(
            f"- `{row['run']}`: `{row['model_type']}`, visual `{row['visual_encoder']}`, "
            f"text `{row['text_encoder']}`, fusion `{row['fusion_mode']}`, HH `{row['hh_drop_mode']}`. "
            f"Per-image rows: `{row['per_image_csv']}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate QaTa checkpoints with per-image and global Dice/IoU")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-json",
        type=str,
        default="qata_dual_metrics_selected.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="qata_dual_metrics_selected.md",
    )
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    failures = []
    for run in args.runs:
        run_dir = Path(run)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        print(f"Evaluating {run_dir.name} on {device}...", flush=True)
        try:
            result = evaluate_run(
                run_dir=run_dir,
                device=device,
                batch_size=max(1, args.batch_size),
                num_workers=max(0, args.num_workers),
                use_amp=args.use_amp,
            )
        except Exception as exc:
            failure = {"run": run_dir.name, "error": repr(exc)}
            failures.append(failure)
            print(f"  FAILED: {failure['error']}", flush=True)
            if args.continue_on_fail:
                continue
            raise
        results.append(result)
        print(
            f"  per-image Dice={result['per_image_dice']:.6f} "
            f"global Dice={result['global_dice']:.6f} "
            f"per-image IoU={result['per_image_iou']:.6f} "
            f"global IoU={result['global_iou']:.6f}",
            flush=True,
        )

    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, output_md)
    if failures:
        failure_path = output_json.with_name(output_json.stem + "_failures.json")
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote {failure_path}", flush=True)
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
