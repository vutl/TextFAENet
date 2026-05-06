from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mosmed_text.py"


def build_command(common_args: argparse.Namespace, dataset_cfg: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        dataset_cfg["data_root"],
        "--dataset-format",
        dataset_cfg["dataset_format"],
        "--save-dir",
        dataset_cfg["save_dir"],
        "--model-type",
        common_args.model_type,
        "--epochs",
        str(common_args.epochs),
        "--early-stop-patience",
        str(common_args.early_stop_patience),
        "--batch-size",
        str(common_args.batch_size),
        "--num-workers",
        str(common_args.num_workers),
        "--image-size",
        str(common_args.image_size),
        "--optimizer",
        common_args.optimizer,
        "--lr",
        str(common_args.lr),
        "--min-lr",
        str(common_args.min_lr),
        "--lr-scheduler",
        common_args.lr_scheduler,
        "--weight-decay",
        str(common_args.weight_decay),
        "--seed",
        str(common_args.seed),
        "--max-text-len",
        str(common_args.max_text_len),
        "--bce-weight",
        str(common_args.bce_weight),
        "--dice-weight",
        str(common_args.dice_weight),
        "--pos-weight",
        common_args.pos_weight,
        "--max-pos-weight",
        str(common_args.max_pos_weight),
        "--metric-thresholds",
        common_args.metric_thresholds,
        "--prompt-mode",
        common_args.prompt_mode,
        "--cxr-bert-dir",
        common_args.cxr_bert_dir,
        "--text-dim",
        str(common_args.text_dim),
        "--fusion-mode",
        common_args.fusion_mode,
        "--hh-drop-mode",
        common_args.hh_drop_mode,
        "--unfreeze-last-n",
        str(common_args.unfreeze_last_n),
        "--lora-r",
        str(common_args.lora_r),
        "--grad-accum-steps",
        str(common_args.grad_accum_steps),
        "--vocab-size",
        str(common_args.vocab_size),
        "--aux-w-d4",
        str(common_args.aux_w_d4),
        "--aux-w-d3",
        str(common_args.aux_w_d3),
        "--aux-w-d2",
        str(common_args.aux_w_d2),
        "--low-level-hf-scale",
        str(common_args.low_level_hf_scale),
        "--spatial-sharpen-power",
        str(common_args.spatial_sharpen_power),
    ]

    cmd.append("--use-cxr-bert" if common_args.use_cxr_bert else "--no-use-cxr-bert")
    cmd.append("--freeze-text-backbone" if common_args.freeze_text_backbone else "--no-freeze-text-backbone")
    if common_args.use_deep_supervision:
        cmd.append("--use-deep-supervision")
    else:
        cmd.append("--no-use-deep-supervision")
    if common_args.use_amp:
        cmd.append("--use-amp")
    else:
        cmd.append("--no-use-amp")
    if common_args.save_debug_vis:
        cmd.extend(["--save-debug-vis", "--debug-vis-samples", str(common_args.debug_vis_samples)])
    else:
        cmd.append("--no-save-debug-vis")

    last_ckpt = Path(dataset_cfg["save_dir"]) / "last.pt"
    if common_args.resume_existing and last_ckpt.exists():
        cmd.extend(["--resume-ckpt", str(last_ckpt)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser("Run text-FAENet training sequentially on MosMed, brain_tumors, and breast_tumors")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lfaenet_tgfs", "lfaenet_tgfs_v2"],
        default="lfaenet_tgfs_v2",
    )
    parser.add_argument("--run-suffix", type=str, default="_fix")
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, choices=["poly", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=str, default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=64.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["native", "canonical", "generic", "lesion", "empty", "shuffle"],
        default="native",
    )
    parser.add_argument("--use-cxr-bert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-text-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--cxr-bert-dir", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--drop-hh-in-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hh-drop-mode", type=str, choices=["zero", "keep", "learned"], default=None)
    parser.add_argument("--fusion-mode", type=str, choices=["encoder", "decoder", "both"], default="decoder")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--use-deep-supervision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-debug-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-vis-samples", type=int, default=8)
    parser.add_argument("--aux-w-d4", type=float, default=0.4)
    parser.add_argument("--aux-w-d3", type=float, default=0.6)
    parser.add_argument("--aux-w-d2", type=float, default=0.8)
    parser.add_argument("--low-level-hf-scale", type=float, default=0.6)
    parser.add_argument("--spatial-sharpen-power", type=float, default=2.0)
    args = parser.parse_args()
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.unfreeze_last_n = max(0, int(args.unfreeze_last_n))
    args.lora_r = max(0, int(args.lora_r))
    args.debug_vis_samples = max(1, int(args.debug_vis_samples))
    if args.hh_drop_mode is None:
        args.hh_drop_mode = "zero" if args.drop_hh_in_decoder else "keep"
    else:
        args.drop_hh_in_decoder = args.hh_drop_mode == "zero"
    if args.model_type != "lfaenet_tgfs_v2" and args.hh_drop_mode == "learned":
        raise ValueError("--hh-drop-mode learned is only implemented for --model-type lfaenet_tgfs_v2")
    if args.model_type != "lfaenet_tgfs_v2" and (args.unfreeze_last_n > 0 or args.lora_r > 0):
        raise ValueError("--unfreeze-last-n and --lora-r are only implemented for --model-type lfaenet_tgfs_v2")
    if not args.use_cxr_bert and (args.unfreeze_last_n > 0 or args.lora_r > 0):
        raise ValueError("--unfreeze-last-n/--lora-r require --use-cxr-bert")
    if args.spatial_sharpen_power <= 0:
        raise ValueError("--spatial-sharpen-power must be positive")

    dataset_sequence = [
        {
            "name": "mosmed",
            "data_root": str(ROOT / "datasets" / "MosMed"),
            "dataset_format": "text_csv",
            "save_dir": str(ROOT / "runs" / f"mosmed_text_faenet_v2_suite{args.run_suffix}"),
        },
        {
            "name": "brain_tumors",
            "data_root": str(ROOT / "datasets" / "brain_tumors"),
            "dataset_format": "prompt_folder",
            "save_dir": str(ROOT / "runs" / f"brain_tumors_text_faenet_v2_suite{args.run_suffix}"),
        },
        {
            "name": "breast_tumors",
            "data_root": str(ROOT / "datasets" / "breast_tumors"),
            "dataset_format": "prompt_folder",
            "save_dir": str(ROOT / "runs" / f"breast_tumors_text_faenet_v2_suite{args.run_suffix}"),
        },
    ]

    for cfg in dataset_sequence:
        cmd = build_command(args, cfg)
        print(f"\n=== Training {cfg['name']} sequentially ===")
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"Training failed for {cfg['name']} with exit code {completed.returncode}")

    print("\nAll datasets completed successfully.")


if __name__ == "__main__":
    main()
