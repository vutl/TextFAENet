from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_QATA = ROOT / "scripts" / "train_qata.py"
QATA_ROOT = r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2"
THRESHOLDS = "0.35,0.40,0.45,0.50,0.55"
CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
CXR_BERT = "BiomedVLP-CXR-BERT-specialized"


VARIANTS: dict[str, dict[str, str | int | float | bool]] = {
    # Strictly comparable with the existing ResNet50+CXR-BERT decoder run:
    # same ResNet50 TGFS-v2 decoder, only the text backbone changes.
    "clinicalbert_v2_decoder_keep": {
        "model_type": "resnet50_tgfs_v2",
        "fusion_mode": "decoder",
        "lr": 0.003,
    },
    # v3 decoder keeps ResNet50 and TGFS decoder but enables the newer
    # learnable frequency priors and multi-head grounding path.
    "clinicalbert_v3_decoder_keep": {
        "model_type": "lfaenet_tgfs_v3",
        "fusion_mode": "decoder",
        "lr": 0.003,
        "grounding_n_heads": 4,
        "learnable_low_level_hf_scale": True,
        "learnable_spatial_sharpen": True,
    },
    # Closest to "use BERT like BERT": token-level cross-attention in the
    # ResNet encoder/bottleneck plus TGFS in the decoder.
    "clinicalbert_v3_both_crossattn_keep": {
        "model_type": "lfaenet_tgfs_v3",
        "fusion_mode": "both",
        "encoder_text_fusion": "cross_attn",
        "lr": 0.003,
        "grounding_n_heads": 4,
        "learnable_low_level_hf_scale": True,
        "learnable_spatial_sharpen": True,
    },
    # Same improved BERT usage, but with the local CXR-BERT checkpoint. This is
    # the direct "main BERT" comparison against the existing CXR-BERT runs.
    "cxrbert_v3_both_crossattn_keep": {
        "model_type": "lfaenet_tgfs_v3",
        "fusion_mode": "both",
        "encoder_text_fusion": "cross_attn",
        "lr": 0.003,
        "text_backbone": CXR_BERT,
        "grounding_n_heads": 4,
        "learnable_low_level_hf_scale": True,
        "learnable_spatial_sharpen": True,
    },
    # Optional lower-risk adapter run. Use a smaller LR because LoRA params and
    # the segmentation model currently share one optimizer group.
    "clinicalbert_v3_both_crossattn_lora4_keep": {
        "model_type": "lfaenet_tgfs_v3",
        "fusion_mode": "both",
        "encoder_text_fusion": "cross_attn",
        "lr": 0.001,
        "lora_r": 4,
        "grounding_n_heads": 4,
        "learnable_low_level_hf_scale": True,
        "learnable_spatial_sharpen": True,
    },
}


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def final_test_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import json
        import math

        data = json.loads(path.read_text(encoding="utf-8"))
        dice = float(data.get("dice", float("nan")))
        loss = float(data.get("loss", 0.0))
        return math.isfinite(dice) and math.isfinite(loss) and dice > 0.0
    except Exception:
        return False


def make_cmd(args: argparse.Namespace, variant_name: str, variant: dict[str, str | int | float | bool]) -> list[str]:
    save_dir = ROOT / "runs" / f"{args.run_prefix}_qata_{variant_name}_seed{args.seed}"
    text_backbone = str(variant.get("text_backbone", args.text_backbone))
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_QATA),
        "--data-root",
        args.qata_data_root,
        "--save-dir",
        str(save_dir),
        "--model-type",
        str(variant["model_type"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--optimizer",
        "sgd",
        "--lr",
        str(float(variant.get("lr", args.lr))),
        "--lr-scheduler",
        "poly",
        "--weight-decay",
        "1e-4",
        "--seed",
        str(args.seed),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--metric-thresholds",
        THRESHOLDS,
        "--save-last-every",
        str(args.save_last_every),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--no-save-best-optimizer",
        "--no-use-deep-supervision",
        "--no-use-amp",
        "--use-cxr-bert",
        "--cxr-bert-dir",
        text_backbone,
        "--text-pooling",
        args.text_pooling,
        "--prompt-mode",
        "native",
        "--hh-drop-mode",
        "keep",
        "--fusion-mode",
        str(variant.get("fusion_mode", "decoder")),
        "--freeze-text-backbone",
        "--unfreeze-last-n",
        str(int(variant.get("unfreeze_last_n", 0))),
        "--lora-r",
        str(int(variant.get("lora_r", 0))),
        "--text-dim",
        "256",
        "--max-text-len",
        "64",
    ]
    if variant["model_type"] == "resnet50_tgfs_v2":
        cmd.extend(["--visual-pretrained", "imagenet"])
    if variant["model_type"] == "lfaenet_tgfs_v3":
        cmd.extend([
            "--v3-encoder-type",
            "resnet50",
            "--pretrained-image-encoder",
            "--freeze-encoder-bn",
            "--encoder-text-fusion",
            str(variant.get("encoder_text_fusion", "film")),
            "--grounding-n-heads",
            str(int(variant.get("grounding_n_heads", 1))),
        ])
        if bool(variant.get("learnable_low_level_hf_scale", False)):
            cmd.append("--learnable-low-level-hf-scale")
        if bool(variant.get("learnable_spatial_sharpen", False)):
            cmd.append("--learnable-spatial-sharpen")
    if args.resume_existing:
        last_ckpt = save_dir / "last.pt"
        if last_ckpt.exists():
            cmd.extend(["--resume-ckpt", str(last_ckpt)])
    return cmd


def run_one(name: str, cmd: list[str], env: dict[str, str], log_dir: Path, dry_run: bool, skip_completed: bool) -> int:
    save_dir = Path(cmd[cmd.index("--save-dir") + 1])
    final_test = save_dir / "final_test.json"
    log_path = log_dir / f"{safe_name(name)}.log"
    if skip_completed and final_test_is_valid(final_test):
        print(f"SKIP completed: {name} ({final_test})", flush=True)
        return 0

    print(f"\n=== {name} ===", flush=True)
    print(quote_cmd(cmd), flush=True)
    with (log_dir / "commands.txt").open("a", encoding="utf-8") as file:
        file.write(f"\n=== {name} ===\n{quote_cmd(cmd)}\n")
    if dry_run:
        return 0

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"=== {name} ===\n{quote_cmd(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser("Run targeted QaTa ResNet + BERT text-backbone experiments.")
    parser.add_argument("--run-prefix", type=str, default="qata_clinicalbert0628")
    parser.add_argument("--text-backbone", type=str, default=CLINICAL_BERT)
    parser.add_argument("--text-pooling", type=str, choices=["mean", "cls", "attn"], default="attn")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=[
        "clinicalbert_v2_decoder_keep",
        "clinicalbert_v3_decoder_keep",
        "clinicalbert_v3_both_crossattn_keep",
        "cxrbert_v3_both_crossattn_keep",
    ])
    parser.add_argument("--include-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qata-data-root", type=str, default=QATA_ROOT)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    variants = list(args.variants)
    if args.include_lora and "clinicalbert_v3_both_crossattn_lora4_keep" not in variants:
        variants.append("clinicalbert_v3_both_crossattn_lora4_keep")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

    batch_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "runs" / "batch_logs" / f"{args.run_prefix}_{batch_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for variant_name in variants:
        cmd = make_cmd(args, variant_name, VARIANTS[variant_name])
        rc = run_one(
            name=f"qata {variant_name} seed={args.seed}",
            cmd=cmd,
            env=env,
            log_dir=log_dir,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
        )
        if rc != 0:
            message = f"qata {variant_name} failed with exit code {rc}"
            print(message, flush=True)
            with (log_dir / "failures.txt").open("a", encoding="utf-8") as file:
                file.write(message + "\n")
            if not args.continue_on_fail:
                raise SystemExit(message)

    print(f"\nLogs saved to: {log_dir}", flush=True)
    print("ClinicalBERT suite completed." if not args.dry_run else "Dry run completed.", flush=True)


if __name__ == "__main__":
    main()
