from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

from scripts.train_brain_tumors import (  # noqa: E402
    CsvPromptedFolderSegmentationDataset,
    SegLoss,
    TextSegCollator,
    _swap_lr_in_text,
    create_model,
    forward_model,
    set_seed,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def namespace_from_config(config: dict, use_amp: bool) -> argparse.Namespace:
    resolved = dict(config)
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


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_test_loader(args: argparse.Namespace, tokenizer, batch_size: int, num_workers: int, device: torch.device):
    dataset = CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="test",
        image_size=int(args.image_size),
        max_samples=getattr(args, "max_test_samples", None),
        csv_path=getattr(args, "test_csv_path", None),
    )
    collator = TextSegCollator(
        tokenizer=tokenizer,
        max_length=int(args.max_text_len),
        prompt_source=str(getattr(args, "prompt_source", "csv")),
        fixed_prompt=str(getattr(args, "fixed_prompt", "Segment the tumor region.")),
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
    return dataset, loader


@torch.inference_mode()
def logits_with_optional_tta(batch, model, args, device, tokenizer):
    mask, logits_orig, _ = forward_model(batch, model, args, device)
    if not bool(getattr(args, "use_tta", False)):
        return mask, logits_orig.float()

    probs_orig = torch.sigmoid(logits_orig.float())
    flipped_batch = {
        "image": torch.flip(batch["image"], dims=[-1]),
        "mask": batch["mask"],
        "mask_name": batch.get("mask_name", []),
    }
    if tokenizer is not None and "input_ids" in batch:
        flipped_texts = [_swap_lr_in_text(str(t)) for t in batch.get("text", [])]
        toks = tokenizer(
            flipped_texts,
            padding="max_length",
            truncation=True,
            max_length=int(args.max_text_len),
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
    probs_flip = torch.sigmoid(torch.flip(logits_flip.float(), dims=[-1]))
    avg_probs = (0.5 * (probs_orig + probs_flip)).clamp(1e-7, 1.0 - 1e-7)
    return mask, torch.log(avg_probs / (1.0 - avg_probs))


@torch.inference_mode()
def evaluate_run(run_dir: Path, device: torch.device, batch_size: int, num_workers: int, use_amp: bool) -> dict:
    config = read_json(run_dir / "config.json")
    args = namespace_from_config(config, use_amp=use_amp)
    set_seed(int(args.seed))

    model, tokenizer = create_model(args, device)
    ckpt = load_checkpoint(checkpoint_path(run_dir), device)
    state = ckpt.get("model_state", ckpt)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("text_encoder.model.")]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {run_dir.name}: missing={missing[:10]} unexpected={unexpected[:10]}")
    model.eval()

    final = read_json(run_dir / "final_test.json") if (run_dir / "final_test.json").exists() else {}
    threshold = float(final.get("best_threshold", ckpt.get("best_threshold", 0.5)))
    _, loader = build_test_loader(args, tokenizer, batch_size, num_workers, device)
    criterion = SegLoss(
        bce_weight=float(getattr(args, "bce_weight", 0.2)),
        dice_weight=float(getattr(args, "dice_weight", 0.8)),
        pos_weight=None,
        boundary_weight=float(getattr(args, "boundary_weight", 0.0)),
    )

    rows: list[dict[str, float | str]] = []
    total_loss = 0.0
    total_intersection = total_union = total_pred = total_target = 0.0
    eps = 1e-6

    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.use_amp)):
            targets, logits = logits_with_optional_tta(batch, model, args, device, tokenizer)
            loss = criterion(logits, targets.to(device).float()).item()

        targets = targets.to(device).float()
        preds = (torch.sigmoid(logits.float()) > threshold).float()
        inter = (preds * targets).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        target_sum = targets.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - inter
        dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
        iou = (inter + eps) / (union + eps)

        names = batch.get("mask_name", [""] * int(targets.shape[0]))
        texts = batch.get("text", [""] * int(targets.shape[0]))
        for idx, name in enumerate(names):
            rows.append(
                {
                    "name": str(name),
                    "dice": float(dice[idx].detach().cpu()),
                    "iou": float(iou[idx].detach().cpu()),
                    "pred_sum": float(pred_sum[idx].detach().cpu()),
                    "target_sum": float(target_sum[idx].detach().cpu()),
                    "gt_area_ratio": float(target_sum[idx].detach().cpu()) / float(targets.shape[-1] * targets.shape[-2]),
                    "text": str(texts[idx]),
                }
            )
        total_loss += loss * int(targets.shape[0])
        total_intersection += float(inter.sum().detach().cpu())
        total_union += float(union.sum().detach().cpu())
        total_pred += float(pred_sum.sum().detach().cpu())
        total_target += float(target_sum.sum().detach().cpu())

    out_csv = run_dir / "test_per_image_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "dice", "iou", "pred_sum", "target_sum", "gt_area_ratio", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    per_image_dice = sum(float(r["dice"]) for r in rows) / max(len(rows), 1)
    per_image_iou = sum(float(r["iou"]) for r in rows) / max(len(rows), 1)
    summary = {
        "run": run_dir.name,
        "threshold": threshold,
        "num_images": len(rows),
        "loss": total_loss / max(len(rows), 1),
        "per_image_dice": per_image_dice,
        "per_image_iou": per_image_iou,
        "global_dice": (2.0 * total_intersection + eps) / (total_pred + total_target + eps),
        "global_iou": (total_intersection + eps) / (total_union + eps),
        "per_image_csv": str(out_csv.relative_to(ROOT)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser("Write per-image metrics for Brain/Breast Text-FAENet runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=str, default="brain_breast_per_image_eval_20260629.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda":
        args.use_amp = False
    results = []
    failures = []
    for run in args.runs:
        run_dir = Path(run)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        try:
            print(f"Evaluating {run_dir.name} on {device}...", flush=True)
            result = evaluate_run(run_dir, device, args.batch_size, args.num_workers, args.use_amp)
            print(
                f"  per-image Dice={result['per_image_dice']:.6f} "
                f"global Dice={result['global_dice']:.6f}",
                flush=True,
            )
            results.append(result)
        except Exception as exc:
            print(f"  FAILED: {exc!r}", flush=True)
            failures.append({"run": run, "error": repr(exc)})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.continue_on_fail:
                raise

    output_path = ROOT / args.output_json
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    if failures:
        fail_path = output_path.with_name(output_path.stem + "_failures.json")
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Wrote {fail_path}")


if __name__ == "__main__":
    main()
