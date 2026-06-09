"""Evaluate a trained MosMed checkpoint (eval-only, no training).

Reuses model/tokenizer/data loading from `train_mosmed_text.py`, so the model
is rebuilt EXACTLY as it was trained.

Loads best.pt + config.json from a runs/<exp> directory. Computes h-flip TTA
(original + flip/swap-text averaged); optionally adds extra scales via --scales
for multi-scale TTA. Caches probabilities once, then sweeps thresholds cheaply.
Reports BOTH global Dice (primary metric) and per-image Dice (reference).

Examples:
    # Test set, checkpoint's best threshold, base scale only (h-flip TTA)
    python scripts/eval_mosmed.py --run-dir runs/mosmed_v9b

    # Multi-scale TTA: also predict at extra scales then average
    python scripts/eval_mosmed.py --run-dir runs/mosmed_v9b --scales 288,320,352,384

    # Sweep thresholds on the val split
    python scripts/eval_mosmed.py --run-dir runs/mosmed_v9b --split val --sweep

    # Force a specific threshold on the test set
    python scripts/eval_mosmed.py --run-dir runs/mosmed_v9b --threshold 0.50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve()
TEXTFAENET_ROOT = ROOT.parents[1]
if str(TEXTFAENET_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXTFAENET_ROOT))

from src.data import MosMedTextCSVDataset


def _load_train_module():
    """Import scripts/train_mosmed_text.py as a module to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(
        "train_mosmed_text",
        str(TEXTFAENET_ROOT / "scripts" / "train_mosmed_text.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_mosmed_text"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_args_from_config(config: dict, cli_overrides) -> object:
    class Ns:
        pass

    args = Ns()
    for k, v in config.items():
        setattr(args, k, v)
    if cli_overrides.batch_size is not None:
        args.batch_size = cli_overrides.batch_size
    if cli_overrides.num_workers is not None:
        args.num_workers = cli_overrides.num_workers
    # Safety defaults for keys that older configs might not have.
    args.max_text_len = getattr(args, "max_text_len", 64)
    args.bce_weight = getattr(args, "bce_weight", 0.2)
    args.dice_weight = getattr(args, "dice_weight", 0.8)
    args.boundary_weight = getattr(args, "boundary_weight", 0.0)
    args.prompt_mode = getattr(args, "prompt_mode", "native")
    args.use_deep_supervision = getattr(args, "use_deep_supervision", True)
    args.grounding_loss_weight = getattr(args, "grounding_loss_weight", 0.0)
    args.aux_w_d4 = getattr(args, "aux_w_d4", 0.4)
    args.aux_w_d3 = getattr(args, "aux_w_d3", 0.6)
    args.aux_w_d2 = getattr(args, "aux_w_d2", 0.8)
    args.metric_thresholds = getattr(args, "metric_thresholds", "0.35,0.40,0.45,0.50,0.55")
    return args


@torch.no_grad()
def multiscale_tta_probs(model, batch, tm, tokenizer, args, device, scales, base_size):
    """Average sigmoid probs over {each scale} x {orig, h-flip}, resampled to base_size."""
    image = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)

    # Flipped-text tokens (l<->r swap) reused across all scales.
    texts = [tm._swap_lr_in_text(str(t)) for t in batch.get("text", [])]
    toks = tokenizer(texts, padding="max_length", truncation=True,
                     max_length=args.max_text_len, return_tensors="pt")
    f_ids = toks["input_ids"].to(device)
    f_attn = toks["attention_mask"].to(device)

    acc = None
    n = 0
    for s in scales:
        img_s = image if s == base_size else F.interpolate(
            image, size=(s, s), mode="bilinear", align_corners=False)
        p = torch.sigmoid(model(img_s, token_ids=input_ids, attention_mask=attn))
        if p.shape[-1] != base_size:
            p = F.interpolate(p, size=(base_size, base_size), mode="bilinear", align_corners=False)
        img_f = torch.flip(img_s, dims=[-1])
        p_f = torch.flip(torch.sigmoid(model(img_f, token_ids=f_ids, attention_mask=f_attn)), dims=[-1])
        if p_f.shape[-1] != base_size:
            p_f = F.interpolate(p_f, size=(base_size, base_size), mode="bilinear", align_corners=False)
        acc = (p + p_f) if acc is None else acc + p + p_f
        n += 2
    return acc / n


def main() -> None:
    parser = argparse.ArgumentParser("Eval-only for a MosMed text-FAENet checkpoint")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="runs/<exp> dir (must contain config.json + the checkpoint)")
    parser.add_argument("--ckpt", type=str, default="best.pt")
    parser.add_argument("--split", type=str, choices=["test", "val", "train"], default="test")
    parser.add_argument("--scales", type=str, default=None,
                        help="Comma-separated extra scales for multi-scale TTA, e.g. '288,320,352,384'. "
                             "Base scale (config.image_size) is always included. "
                             "Default: base scale only (h-flip TTA only).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold to apply. Default: checkpoint's best_threshold.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all config thresholds and print a table.")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA entirely (no h-flip, no extra scales).")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    cli_args = parser.parse_args()

    run_dir = Path(cli_args.run_dir)
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / cli_args.ckpt
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {cli_args.ckpt} at {ckpt_path}")

    print(f"Loading config from {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = _build_args_from_config(config, cli_args)

    tm = _load_train_module()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda"
    print(f"Device: {device}")

    base_size = int(args.image_size)
    ds = MosMedTextCSVDataset(
        root_dir=args.data_root,
        split=cli_args.split,
        image_size=base_size,
        max_samples=cli_args.max_samples,
        augment=False,
        ct_window=bool(getattr(args, "ct_window", False)),
    )
    print(f"{cli_args.split} samples: {len(ds)}  ct_window={getattr(args, 'ct_window', False)}")

    # Build scale list: always include base_size; append any extra from --scales.
    if cli_args.no_tta:
        scales = None
    else:
        scales = [base_size]
        if cli_args.scales:
            for s in [int(x) for x in cli_args.scales.split(",")]:
                if s not in scales:
                    scales.append(s)
    print(f"TTA scales: {'disabled' if scales is None else scales}")

    model, tokenizer = tm.create_model(args, device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = tm.load_checkpoint(ckpt_path, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  loaded. missing keys (expected for frozen text backbone): {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} — first few: {unexpected[:3]}")
    ckpt_threshold = float(ckpt.get("best_threshold", 0.5))
    best_epoch = int(ckpt.get("epoch", -1))
    print(f"  best_epoch={best_epoch}  ckpt_threshold={ckpt_threshold}")

    collate_fn = tm.TextSegCollator(
        tokenizer=tokenizer,
        max_length=int(args.max_text_len),
        prompt_mode=args.prompt_mode,
    )
    loader = DataLoader(
        ds, batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Compute probabilities once, cache on CPU, then sweep thresholds cheaply.
    from tqdm import tqdm
    all_probs, all_gts = [], []
    model.eval()
    for batch in tqdm(loader, desc=f"forward {cli_args.split}"):
        gt = batch["mask"]
        if scales is None:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(image, token_ids=input_ids, attention_mask=attn))
        else:
            with torch.no_grad():
                p = multiscale_tta_probs(model, batch, tm, tokenizer, args, device, scales, base_size)
        all_probs.append(p.cpu())
        all_gts.append(gt.cpu())

    all_probs_t = torch.cat(all_probs, dim=0)   # N, 1, H, W
    all_gts_t = torch.cat(all_gts, dim=0)
    N = all_probs_t.shape[0]
    print(f"Cached probs for {N} images.")

    eps = 1e-6
    applied_threshold = cli_args.threshold if cli_args.threshold is not None else ckpt_threshold
    thresholds_to_sweep = (
        tm.parse_thresholds(args.metric_thresholds) if cli_args.sweep else [applied_threshold]
    )

    results = {}
    for t in thresholds_to_sweep:
        pred = (all_probs_t > t).float()
        inter = (pred * all_gts_t).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        gt_sum = all_gts_t.sum(dim=(1, 2, 3))
        union = ((pred + all_gts_t) > 0).float().sum(dim=(1, 2, 3))
        results[t] = {
            "global_dice": (2 * inter.sum().item() + eps) / (pred_sum.sum().item() + gt_sum.sum().item() + eps),
            "global_iou": (inter.sum().item() + eps) / (union.sum().item() + eps),
            "per_img_dice": ((2 * inter + eps) / (pred_sum + gt_sum + eps)).mean().item(),
            "per_img_iou": ((inter + eps) / (union + eps)).mean().item(),
        }

    if cli_args.sweep:
        print(f"\n{'thr':>5} | {'global Dice':>11} {'global IoU':>10} | {'per-img Dice':>12} {'per-img IoU':>11}")
        print("-" * 64)
        for t in thresholds_to_sweep:
            r = results[t]
            print(f"{t:>5.2f} | {r['global_dice']:>11.4f} {r['global_iou']:>10.4f} | "
                  f"{r['per_img_dice']:>12.4f} {r['per_img_iou']:>11.4f}")
        best_t = max(thresholds_to_sweep, key=lambda t: results[t]["global_dice"])
        print(f"\nbest (global Dice): thr={best_t:.2f}  global_dice={results[best_t]['global_dice']:.4f}")

    r = results[applied_threshold]
    summary = {
        "split": cli_args.split,
        "ckpt": cli_args.ckpt,
        "best_epoch": best_epoch,
        "ckpt_threshold": ckpt_threshold,
        "applied_threshold": applied_threshold,
        "scales": scales,
        "global_dice": float(r["global_dice"]),
        "global_iou": float(r["global_iou"]),
        "per_img_dice": float(r["per_img_dice"]),
        "per_img_iou": float(r["per_img_iou"]),
    }
    out_name = cli_args.output_json or f"eval_{cli_args.split}_thr{int(round(applied_threshold * 100)):02d}.json"
    output_path = run_dir / out_name
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
