from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mosmed_text.py"


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
    ]
    return cmd


def _cfg_m0(args: argparse.Namespace) -> list[str]:
    return [
        "--model-type",
        "lfaenet_tgfs_v2",
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
    base = _base_cmd(args, case_id)

    # ── M10 family: Hybrid raw HF/LF dual-branch + TGFS decoder ──────────────
    # All four variants share IDENTICAL M6 hyperparameters so the ONLY variable
    # is how DWT input is handled.  This is a strict ablation.
    #
    #  M10a  upsample               — DWT then bilinear upsample back to H×W
    #  M10b  pad_crop               — reflect-pad before every haar_dwt2d, crop after iDWT
    #  M10c  lowres_conv_bottleneck — encoders at H/2, plain conv bottleneck
    #  M10d  lowres_pad_crop        — encoders at H/2, pad-crop + plain conv bottleneck
    #  M10e  stem_dwt               — shared full-res stem, DWT on features, dec0 at H×W
    #                                 (recommended: no bilinear upsample of logits)

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
        # batch_size=2 + grad_accum=4 keeps effective batch=8 but saves VRAM
        "--batch-size",        "2",
        "--grad-accum-steps",  "4",
    ]

    if case_id == "M10E":
        return base + _M10_SHARED + ["--dwt-strategy", "stem_dwt"], \
            "M10E: Stem-DWT (shared full-res stem, DWT on features, dec0 at H×W — no lossy upsample)"

    if case_id == "M10A":
        return base + _M10_SHARED + ["--dwt-strategy", "upsample"], \
            "M10A: DWT→upsample (encoders at H×W; same spatial path as V2)"

    if case_id == "M10B":
        return base + _M10_SHARED + ["--dwt-strategy", "pad_crop"], \
            "M10B: DWT→pad/crop (encoders at H/2; FreqASafe everywhere)"

    if case_id == "M10C":
        return base + _M10_SHARED + ["--dwt-strategy", "lowres_conv_bottleneck"], \
            "M10C: DWT→lowres+conv-bottleneck (encoders at H/2; plain conv at bottleneck, requires even input)"

    if case_id == "M10D":
        return base + _M10_SHARED + ["--dwt-strategy", "lowres_pad_crop"], \
            "M10D: DWT→lowres+pad/crop+conv-bottleneck (recommended; closest to original plan)"


    if case_id == "M0":
        return base + _cfg_m0(args), "Baseline: decoder + CXR-BERT frozen + hard priors"

    if case_id == "M1":
        return base + [
            "--model-type",
            "faenet",
            "--no-text",
            "--prompt-mode",
            "empty",
        ], "Visual-only FAENet"

    if case_id == "M2":
        return base + _cfg_m0(args) + ["--prompt-mode", "shuffle"], "Text branch bypass test (shuffle)"

    if case_id == "M3":
        # Using empty prompt as primary M3 proxy (text presence vs identity).
        return base + _cfg_m0(args) + ["--prompt-mode", "empty"], "Text identity vs presence (empty prompt)"

    if case_id == "M4":
        # Canonical policy as default; you can rerun M4 with lesion prompt if needed.
        return base + _cfg_m0(args) + ["--prompt-mode", "canonical"], "Prompt policy test (canonical)"

    if case_id == "M5":
        return base + _cfg_m0(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
        ], "Fusion position test (both)"

    if case_id == "M6":
        return base + _cfg_m0(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
            "--no-use-cxr-bert",
        ], "Universal text encoder test (simple encoder)"

    if case_id == "M7":
        if args.m7_mode == "lora":
            return base + _cfg_m0(args) + [
                "--fusion-mode",
                "both",
                "--prompt-mode",
                "native",
                "--lora-r",
                str(args.lora_r),
                "--lora-alpha",
                str(args.lora_alpha),
            ], "Adaptive text encoder test (CXR-BERT + LoRA)"
        return base + _cfg_m0(args) + [
            "--fusion-mode",
            "both",
            "--prompt-mode",
            "native",
            "--no-freeze-text-backbone",
            "--unfreeze-last-n",
            str(args.unfreeze_last_n),
        ], "Adaptive text encoder test (partial unfreeze)"

    if case_id == "M8":
        return base + _cfg_m0(args) + [
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
        # Phase 2 confirmed M6 as the best config. M9 = M6 + deep supervision.
        # Explicitly building the command to avoid conflicting overrides from M0.
        return base + [
            "--model-type",
            "lfaenet_tgfs_v2",
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
        return base + [
            "--model-type", "lfaenet_tgfs_v2",
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
    parser.add_argument("--save-root", type=str, default=str(ROOT / "runs" / "mosmed_matrix_m0_m10"))
    parser.add_argument("--cases", type=str, default="M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--metric-thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--m7-mode", choices=["lora", "unfreeze"], default="lora")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    case_ids = parse_cases(args.cases)
    if not case_ids:
        raise ValueError("No valid case ids provided via --cases")

    print(f"Running cases: {', '.join(case_ids)}")
    print(f"Save root: {args.save_root}")

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
