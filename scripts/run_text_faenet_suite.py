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
        "--cxr-bert-dir",
        common_args.cxr_bert_dir,
        "--text-dim",
        str(common_args.text_dim),
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

    if common_args.use_cxr_bert:
        cmd.append("--use-cxr-bert")
    if common_args.freeze_text_backbone:
        cmd.append("--freeze-text-backbone")
    if common_args.drop_hh_in_decoder:
        cmd.append("--drop-hh-in-decoder")
    else:
        cmd.append("--no-drop-hh-in-decoder")
    if common_args.use_deep_supervision:
        cmd.append("--use-deep-supervision")
    else:
        cmd.append("--no-use-deep-supervision")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser("Run text-FAENet training sequentially on MosMed, brain_tumors, and breast_tumors")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lfaenet_tgfs", "lfaenet_tgfs_v2"],
        default="lfaenet_tgfs_v2",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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
    parser.add_argument("--metric-thresholds", type=str, default="0.2,0.3,0.4,0.5")
    parser.add_argument("--use-cxr-bert", action="store_true", default=True)
    parser.add_argument("--freeze-text-backbone", action="store_true", default=True)
    parser.add_argument("--cxr-bert-dir", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--drop-hh-in-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--use-deep-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-w-d4", type=float, default=0.4)
    parser.add_argument("--aux-w-d3", type=float, default=0.6)
    parser.add_argument("--aux-w-d2", type=float, default=0.8)
    parser.add_argument("--low-level-hf-scale", type=float, default=0.6)
    parser.add_argument("--spatial-sharpen-power", type=float, default=2.0)
    args = parser.parse_args()

    dataset_sequence = [
        {
            "name": "mosmed",
            "data_root": str(ROOT / "datasets" / "MosMed"),
            "dataset_format": "text_csv",
            "save_dir": str(ROOT / "runs" / "mosmed_text_faenet_v2_suite"),
        },
        {
            "name": "brain_tumors",
            "data_root": str(ROOT / "datasets" / "brain_tumors"),
            "dataset_format": "prompt_folder",
            "save_dir": str(ROOT / "runs" / "brain_tumors_text_faenet_v2_suite"),
        },
        {
            "name": "breast_tumors",
            "data_root": str(ROOT / "datasets" / "breast_tumors"),
            "dataset_format": "prompt_folder",
            "save_dir": str(ROOT / "runs" / "breast_tumors_text_faenet_v2_suite"),
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
