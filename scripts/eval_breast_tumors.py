from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve()
TEXTFAENET_ROOT = ROOT.parents[1]
if str(TEXTFAENET_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXTFAENET_ROOT))


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_brain_tumors",
        str(TEXTFAENET_ROOT / "scripts" / "train_brain_tumors.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_brain_tumors"] = mod
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
    if cli_overrides.max_test_samples is not None:
        args.max_test_samples = cli_overrides.max_test_samples
    args.max_text_len = getattr(args, "max_text_len", 64)
    args.bce_weight = getattr(args, "bce_weight", 0.3)
    args.dice_weight = getattr(args, "dice_weight", 0.7)
    args.boundary_weight = getattr(args, "boundary_weight", 0.0)
    args.prompt_source = getattr(args, "prompt_source", "csv")
    args.fixed_prompt = getattr(args, "fixed_prompt", "Segment the tumor region.")
    args.norm_type = getattr(args, "norm_type", "bn")
    args.conv_block_depth = getattr(args, "conv_block_depth", 2)
    args.dropout_p = getattr(args, "dropout_p", 0.0)
    args.grounding_n_heads = getattr(args, "grounding_n_heads", 1)
    args.encoder_type = getattr(args, "encoder_type", "from_scratch")
    args.pretrained_image_encoder = getattr(args, "pretrained_image_encoder", True)
    args.freeze_encoder_bn = getattr(args, "freeze_encoder_bn", True)
    args.grounding_loss_weight = getattr(args, "grounding_loss_weight", 0.0)
    return args


def main() -> None:
    parser = argparse.ArgumentParser("Re-evaluate checkpoint with a fixed threshold")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best.pt")
    parser.add_argument("--threshold", type=float, required=True, default=0.55,
                        help="Fixed threshold to apply (overrides ckpt's best_threshold).")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output filename (default: final_test_thr<NN>.json)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable h-flip TTA (default: same TTA used during training)")
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

    tbt = _load_train_module()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda" and args.model_type != "faenet"
    print(f"Device: {device}")

    test_ds = tbt.CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="test",
        image_size=int(args.image_size),
        max_samples=getattr(args, "max_test_samples", None),
    )
    print(f"Test samples: {len(test_ds)}")

    model, tokenizer = tbt.create_model(args, device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = tbt.load_checkpoint(ckpt_path, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  loaded. missing (expected for frozen text backbone): {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} — first few: {unexpected[:3]}")
    ckpt_threshold = float(ckpt.get("best_threshold", 0.5))
    best_epoch = int(ckpt.get("epoch", -1))
    print(f"  best_epoch={best_epoch}")
    print(f"  ckpt_threshold={ckpt_threshold} (will be overridden)")
    print(f"  fixed_threshold={cli_args.threshold}")

    pos_weight = float(getattr(args, "resolved_pos_weight", 1.0))
    criterion = tbt.SegLoss(
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
        pos_weight=pos_weight,
        boundary_weight=float(args.boundary_weight),
    )

    collate_fn = tbt.TextSegCollator(
        tokenizer=tokenizer if args.model_type != "faenet" else None,
        max_length=int(args.max_text_len),
        prompt_source=args.prompt_source,
        fixed_prompt=args.fixed_prompt,
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    use_tta = (not cli_args.no_tta) and bool(getattr(args, "use_tta", False))
    print(f"Running test eval (tta={use_tta}, threshold={cli_args.threshold})...")
    if use_tta:
        test_stats = tbt.run_test_with_tta(
            model, test_loader, criterion, device, args, tokenizer,
            threshold=cli_args.threshold,
        )
    else:
        test_stats = tbt.run_epoch(
            model, test_loader, criterion, device, args,
            optimizer=None, scaler=None, threshold=cli_args.threshold,
        )

    summary = {
        "best_epoch": best_epoch,
        "ckpt_threshold": ckpt_threshold,
        "applied_threshold": cli_args.threshold,
        "tta": use_tta,
        "loss": float(test_stats["loss"]),
        "iou": float(test_stats["iou"]),
        "dice": float(test_stats["dice"]),
        "pred_pos_ratio": float(test_stats["pred_pos_ratio"]),
        "gt_pos_ratio": float(test_stats["gt_pos_ratio"]),
    }
    out_name = cli_args.output_json
    if out_name is None:
        thr_str = f"{int(round(cli_args.threshold * 100)):02d}"
        out_name = f"final_test_thr{thr_str}.json"
    output_path = run_dir / out_name
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
