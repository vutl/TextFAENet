from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mosmed_text.py"
TRAIN_NOTEXT_SCRIPT = ROOT / "scripts" / "train_mosmed.py"


def _quoted(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _base_cmd(args: argparse.Namespace, case_id: str) -> list[str]:
    save_dir = Path(args.save_root) / case_id
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        args.data_root,
        "--dataset-format",
        "text_csv",
        "--save-dir",
        str(save_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--optimizer",
        "adamw",
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--lr-scheduler",
        "cosine",
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--metric-thresholds",
        args.metric_thresholds,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--image-size",
        str(args.image_size),
        "--encoder-type",
        args.encoder_type,
    ]
    # Pretrained encoder flags
    if args.encoder_type == "resnet50":
        cmd += ["--pretrained-image-encoder", "--freeze-encoder-bn"]
        if args.encoder_lr > 0:
            cmd += ["--encoder-lr", str(args.encoder_lr)]
        if args.max_grad_norm > 0:
            cmd += ["--max-grad-norm", str(args.max_grad_norm)]
        if args.optim_eps != 1e-8:
            cmd += ["--optim-eps", str(args.optim_eps)]
        if args.lr_warmup_epochs > 0:
            cmd += ["--lr-warmup-epochs", str(args.lr_warmup_epochs)]
    # Augmentation
    if args.augment_train:
        cmd += ["--augment-train"]
    else:
        cmd += ["--no-augment-train"]
    # Norm type and conv depth
    if args.norm_type != "bn":
        cmd += ["--norm-type", args.norm_type]
    if args.conv_block_depth != 2:
        cmd += ["--conv-block-depth", str(args.conv_block_depth)]
    if args.dropout_p > 0:
        cmd += ["--dropout-p", str(args.dropout_p)]
    # Use global dice for selection
    if args.use_global_dice_selection:
        cmd += ["--use-global-dice-selection"]
    else:
        cmd += ["--no-use-global-dice-selection"]
    return cmd


def _base_notext_cmd(args: argparse.Namespace, case_id: str) -> list[str]:
    """Build base command for the visual-only FAENet script (train_mosmed.py).

    This script uses the 'prepared' dataset format and a separate set of CLI
    flags, so we construct the command from scratch here.
    """
    save_dir = Path(args.save_root) / case_id
    cmd = [
        sys.executable,
        str(TRAIN_NOTEXT_SCRIPT),
        "--prepared-root",
        args.prepared_root,
        "--save-dir",
        str(save_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--lr-scheduler",
        "cosine",
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--metric-thresholds",
        args.metric_thresholds,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--image-size",
        str(args.image_size),
        "--encoder-type",
        args.encoder_type,
    ]
    if args.encoder_type == "resnet50":
        cmd += ["--pretrained-image-encoder", "--freeze-encoder-bn"]
        if args.encoder_lr > 0:
            cmd += ["--encoder-lr", str(args.encoder_lr)]
        if args.max_grad_norm > 0:
            cmd += ["--max-grad-norm", str(args.max_grad_norm)]
        if args.optim_eps != 1e-8:
            cmd += ["--optim-eps", str(args.optim_eps)]
        if args.lr_warmup_epochs > 0:
            cmd += ["--lr-warmup-epochs", str(args.lr_warmup_epochs)]
    return cmd


def _common_text_flags(args: argparse.Namespace) -> list[str]:
    """Common flags shared by all text-model M-cases (M0, M2–M8)."""
    return [
        "--prompt-mode",
        "native",
        "--fusion-mode",
        "decoder",
        "--use-cxr-bert",
        "--freeze-text-backbone",
        "--hh-drop-mode",
        "zero",
        "--low-level-hf-scale",
        "0.6",
        "--spatial-sharpen-power",
        "2.0",
        "--no-use-deep-supervision",
    ]


def build_case_cmd(args: argparse.Namespace, case_id: str) -> tuple[list[str] | None, str]:
    # ── M10 family: Hybrid raw HF/LF dual-branch + TGFS decoder ──────────────
    _M10_SHARED = [
        "--model-type",        "hybrid_v2",
        "--prompt-mode",       "native",
        "--fusion-mode",       "both",
        "--no-use-cxr-bert",
        "--freeze-text-backbone",
        "--hh-drop-mode",      "zero",
        "--low-level-hf-scale","0.6",
        "--spatial-sharpen-power", "2.0",
        "--no-use-deep-supervision",
        "--batch-size",        "2",
        "--grad-accum-steps",  "4",
    ]

    if case_id == "M10E":
        base = _base_cmd(args, case_id)
        return base + _M10_SHARED + ["--dwt-strategy", "stem_dwt"], \
            "M10E: Stem-DWT (shared full-res stem, DWT on features, dec0 at H×W — no lossy upsample)"

    if case_id == "M10A":
        base = _base_cmd(args, case_id)
        return base + _M10_SHARED + ["--dwt-strategy", "upsample"], \
            "M10A: DWT→upsample (encoders at H×W; same spatial path as V2)"

    if case_id == "M10B":
        base = _base_cmd(args, case_id)
        return base + _M10_SHARED + ["--dwt-strategy", "pad_crop"], \
            "M10B: DWT→pad/crop (encoders at H/2; FreqASafe everywhere)"

    if case_id == "M10C":
        base = _base_cmd(args, case_id)
        return base + _M10_SHARED + ["--dwt-strategy", "lowres_conv_bottleneck"], \
            "M10C: DWT→lowres+conv-bottleneck (encoders at H/2; plain conv at bottleneck, requires even input)"

    if case_id == "M10D":
        base = _base_cmd(args, case_id)
        return base + _M10_SHARED + ["--dwt-strategy", "lowres_pad_crop"], \
            "M10D: DWT→lowres+pad/crop+conv-bottleneck (recommended; closest to original plan)"

    if case_id == "M0":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args), "Baseline: decoder + CXR-BERT frozen + hard priors"

    if case_id == "M1":
        # Visual-only FAENet (no text). Now runs inside train_mosmed_text.py
        # using --model-type faenet so it shares the exact same data split as M0.
        base = _base_cmd(args, case_id)
        return base + ["--model-type", "faenet", "--prompt-mode", "empty"], "Visual-only FAENet (no text branch)"

    if case_id == "M2":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + ["--prompt-mode", "shuffle"], "Text branch bypass test (shuffle)"

    if case_id == "M3":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + ["--prompt-mode", "empty"], "Text identity vs presence (empty prompt)"

    if case_id == "M4":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + ["--prompt-mode", "canonical"], "Prompt policy test (canonical)"

    if case_id == "M5":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
        ], "Fusion position test (both)"

    if case_id == "M6":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
            "--no-use-cxr-bert",
        ], "Universal text encoder test (simple encoder)"

    if case_id == "M7":
        base = _base_cmd(args, case_id)
        if args.m7_mode == "lora":
            return base + _common_text_flags(args) + [
                "--fusion-mode",
                "both",
                "--prompt-mode",
                "native",
                "--lora-r",
                str(args.lora_r),
                "--lora-alpha",
                str(args.lora_alpha),
            ], "Adaptive text encoder test (CXR-BERT + LoRA)"
        return base + _common_text_flags(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
            "--no-freeze-text-backbone",
            "--unfreeze-last-n",
            str(args.unfreeze_last_n),
        ], "Adaptive text encoder test (partial unfreeze)"

    if case_id == "M8":
        base = _base_cmd(args, case_id)
        return base + _common_text_flags(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
            "--hh-drop-mode",
            "learned",
            "--low-level-hf-scale",
            "1.0",
            "--learnable-low-level-hf-scale",
            "--spatial-sharpen-power",
            "1.0",
            "--learnable-spatial-sharpen",
        ], "De-biased frequency priors"

    if case_id == "M9":
        base = _base_cmd(args, case_id)
        return base + [
            "--prompt-mode",
            "native",
            "--fusion-mode",
            "both",
            "--no-use-cxr-bert",
            "--freeze-text-backbone",
            "--hh-drop-mode",
            "zero",
            "--low-level-hf-scale",
            "0.6",
            "--spatial-sharpen-power",
            "2.0",
            "--use-deep-supervision",
        ], "Deep supervision on top of M6 (best Phase 2 config)"

    if case_id == "M11":
        base = _base_cmd(args, case_id)
        return base + [
            "--prompt-mode", "native",
            "--fusion-mode", "both",
            "--no-use-deep-supervision",
            "--use-cxr-bert",
            "--no-freeze-text-backbone",
            "--unfreeze-last-n", "2",
            "--hh-drop-mode", "learned",
            "--learnable-low-level-hf-scale",
            "--learnable-spatial-sharpen",
            "--boundary-weight", "0.15",
            "--encoder-text-fusion", "cross_attn",
        ], "M11: M6-Pro (Native Text, CXR-BERT unfrozen, Cross-Attn Encoder, Boundary Loss, Learnable Priors)"

    raise ValueError(f"Unknown case_id: {case_id}")


def parse_cases(spec: str) -> list[str]:
    out: list[str] = []
    for raw in spec.split(","):
        item = raw.strip().upper()
        if item:
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Run MosMed screening matrix M0-M10 sequentially")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/Users/server/Genius_Lab/projects/dataset/COVID_CT_MosMed",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=str(ROOT / "datasets" / "mosmed_2d_prepared"),
        help="Root of the prepared (non-text) MosMed dataset for M1 (visual-only FAENet).",
    )
    parser.add_argument("--save-root", type=str, default=str(ROOT / "runs" / "mosmed_matrix_m0_m10"))
    parser.add_argument(
        "--cases",
        type=str,
        default="M0,M1,M2,M3,M4,M5,M6,M7,M8",
        help="Comma-separated list of cases to run (e.g., M0,M1,M5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--m7-mode", choices=["lora", "unfreeze"], default="lora")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    # New flags for upgraded config
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--encoder-type", type=str, choices=["from_scratch", "resnet50"], default="resnet50")
    parser.add_argument("--encoder-lr", type=float, default=5e-6,
                        help="Separate LR for ResNet-50 encoder backbone.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim-eps", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--augment-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-type", type=str, choices=["bn", "gn"], default="gn")
    parser.add_argument("--conv-block-depth", type=int, choices=[2, 3], default=3)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--use-global-dice-selection", action=argparse.BooleanOptionalAction, default=True,
                        help="Use global (pooled) Dice for checkpoint selection instead of per-image Dice.")
    args = parser.parse_args()

    case_ids = parse_cases(args.cases)
    if not case_ids:
        raise ValueError("No valid case ids provided via --cases")

    print(f"Running cases: {', '.join(case_ids)}")
    print(f"Save root: {args.save_root}")
    print(f"Config: image_size={args.image_size}, encoder={args.encoder_type}, "
          f"epochs={args.epochs}, augment={args.augment_train}, "
          f"norm={args.norm_type}, conv_depth={args.conv_block_depth}")

    for case_id in case_ids:
        cmd, note = build_case_cmd(args, case_id)
        print(f"\n=== {case_id} ===")
        print(f"Note: {note}")

        if cmd is None:
            continue

        print(_quoted(cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"{case_id} failed with exit code {completed.returncode}")

    print("\nDone.")


if __name__ == "__main__":
    main()
