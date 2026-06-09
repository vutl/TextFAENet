"""Re-evaluate a brain_tumors checkpoint with a fixed threshold.

The model config is hardcoded inline (mirrors runs/brain_tumors/config.json).
No external config.json file is needed — just point to a checkpoint and a
threshold. Override knobs are exposed via CLI for batch size, num workers,
TTA, and the test-sample cap.

Example:
    python scripts/eval_brain_tumors.py --ckpt runs/brain_tumors/best.pt --threshold 0.55
    python scripts/eval_brain_tumors.py --run-dir runs/brain_tumors --threshold 0.55
"""

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


# ---------------------------------------------------------------------------
# Hardcoded config (mirrors runs/brain_tumors/config.json from the v7a run).
# Edit values here directly instead of supplying a config.json file.
# ---------------------------------------------------------------------------
CONFIG: dict = {
    "data_root": str(TEXTFAENET_ROOT.parent / "dataset" / "MedCLIP-SAMv2_data" / "brain_tumors"),
    "image_size": 320,
    "batch_size": 2,
    "num_workers": 2,
    "max_text_len": 64,
    "max_test_samples": None,
    "bce_weight": 0.2,
    "dice_weight": 0.8,
    "boundary_weight": 0.1,
    "pos_weight": "auto",
    "max_pos_weight": 16.0,
    "resolved_pos_weight": 16.0,
    "use_cxr_bert": True,
    "cxr_bert_dir": str(TEXTFAENET_ROOT / "BiomedVLP-CXR-BERT-specialized"),
    "freeze_text_backbone": True,
    "drop_hh_in_decoder": None,
    "hh_drop_mode": "learned",
    "unfreeze_last_n": 0,
    "lora_r": 0,
    "lora_alpha": 16.0,
    "fusion_mode": "both",
    "text_dim": 256,
    "vocab_size": 30522,
    "use_deep_supervision": True,
    "aux_w_d4": 0.4,
    "aux_w_d3": 0.6,
    "aux_w_d2": 0.8,
    "low_level_hf_scale": 0.6,
    "learnable_low_level_hf_scale": True,
    "spatial_sharpen_power": 2.0,
    "learnable_spatial_sharpen": True,
    "encoder_text_fusion": "cross_attn",
    "prompt_source": "csv",
    "fixed_prompt": "Segment the tumor region.",
    "augment_train": True,
    "norm_type": "gn",
    "conv_block_depth": 3,
    "dropout_p": 0.1,
    "grounding_n_heads": 4,
    "grounding_loss_weight": 0.3,
    "use_tta": True,
    "encoder_type": "resnet50",
    "pretrained_image_encoder": True,
    "freeze_encoder_bn": True,
    "encoder_lr": 1e-5,
    "model_type": "lfaenet_tgfs_v2",
}


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_brain_tumors",
        str(TEXTFAENET_ROOT / "scripts" / "train_brain_tumors.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_brain_tumors"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_args(cli_overrides) -> object:
    class Ns:
        pass

    args = Ns()
    for k, v in CONFIG.items():
        setattr(args, k, v)
    if cli_overrides.data_root is not None:
        args.data_root = cli_overrides.data_root
    if cli_overrides.batch_size is not None:
        args.batch_size = cli_overrides.batch_size
    if cli_overrides.num_workers is not None:
        args.num_workers = cli_overrides.num_workers
    if cli_overrides.max_test_samples is not None:
        args.max_test_samples = cli_overrides.max_test_samples
    return args


def main() -> None:
    parser = argparse.ArgumentParser("Re-evaluate brain_tumors checkpoint with fixed threshold")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (best.pt). If omitted, --run-dir is used.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to run directory containing best.pt. Output JSON is saved here.")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Fixed threshold applied to sigmoid probs for binary metrics. Default 0.55.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output filename (default: final_test_thr<NN>.json next to ckpt)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data_root from hardcoded config.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable h-flip TTA (default: same TTA used during training)")
    cli_args = parser.parse_args()

    # Resolve checkpoint path: explicit --ckpt, or <run-dir>/best.pt.
    if cli_args.ckpt is not None:
        ckpt_path = Path(cli_args.ckpt)
        run_dir = ckpt_path.parent if cli_args.run_dir is None else Path(cli_args.run_dir)
    elif cli_args.run_dir is not None:
        run_dir = Path(cli_args.run_dir)
        ckpt_path = run_dir / "best.pt"
    else:
        parser.error("Must specify either --ckpt or --run-dir")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    args = _build_args(cli_args)
    tbt = _load_train_module()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.use_amp = device.type == "cuda" and args.model_type != "faenet"
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    test_ds = tbt.CsvPromptedFolderSegmentationDataset(
        root_dir=args.data_root,
        split="test",
        image_size=int(args.image_size),
        max_samples=getattr(args, "max_test_samples", None),
    )
    print(f"Test samples: {len(test_ds)}")

    model, tokenizer = tbt.create_model(args, device)

    ckpt = tbt.load_checkpoint(ckpt_path, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  loaded. missing (expected for frozen text backbone): {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} — first few: {unexpected[:3]}")
    ckpt_threshold = float(ckpt.get("best_threshold", 0.5))
    best_epoch = int(ckpt.get("epoch", -1))
    print(f"  best_epoch={best_epoch}")
    print(f"  ckpt_threshold={ckpt_threshold}")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
