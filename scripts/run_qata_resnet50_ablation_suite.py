from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"


SIMPLE_CORE_VARIANTS = [
    "resnet50_faenet_visual_clean",
    "resnet50_simple_native_keep_decoder",
    "resnet50_simple_empty_keep_decoder",
    "resnet50_simple_shuffle_keep_decoder",
    "resnet50_simple_generic_keep_decoder",
]

SIMPLE_PRIOR_VARIANTS = [
    "resnet50_simple_native_zero_decoder",
    "resnet50_simple_native_learned_decoder",
]

SIMPLE_FREQUENCY_DROP_VARIANTS = [
    "resnet50_simple_native_drop_ll_decoder",
    "resnet50_simple_native_drop_lh_decoder",
    "resnet50_simple_native_drop_hl_decoder",
    "resnet50_simple_native_drop_hh_decoder",
]

CXR_CORE_VARIANTS = [
    "resnet50_cxr_native_keep_decoder",
    "resnet50_cxr_empty_keep_decoder",
    "resnet50_cxr_shuffle_keep_decoder",
    "resnet50_cxr_generic_keep_decoder",
]

CXR_PRIOR_VARIANTS = [
    "resnet50_cxr_native_zero_decoder",
    "resnet50_cxr_native_learned_decoder",
]

CXR_TEXT_ADAPT_VARIANTS = [
    "resnet50_cxr_lora8_keep_decoder",
]

CXR_FREQUENCY_DROP_VARIANTS = [
    "resnet50_cxr_native_drop_ll_decoder",
    "resnet50_cxr_native_drop_lh_decoder",
    "resnet50_cxr_native_drop_hl_decoder",
    "resnet50_cxr_native_drop_hh_decoder",
]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def main() -> None:
    parser = argparse.ArgumentParser("Run the QaTa ResNet-50 TGFS ablation suite sequentially.")
    parser.add_argument("--run-prefix", type=str, default="qata_resnet0523")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["core", "full", "exhaustive", "simple-only", "cxr-only"],
        default="full",
        help=(
            "core: 4 simple prompt sanity runs; full: simple core+priors+frequency drops; "
            "exhaustive: full plus matching CXR-BERT runs and LoRA; simple-only/cxr-only restrict families."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--core-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-frequency-drops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--qata-data-root",
        type=str,
        default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2",
    )
    args = parser.parse_args()

    suite = "core" if args.core_only else args.suite
    variants: list[str] = []
    if suite in {"core", "full", "exhaustive", "simple-only"}:
        variants.extend(SIMPLE_CORE_VARIANTS)
        if suite != "core" and args.include_priors:
            variants.extend(SIMPLE_PRIOR_VARIANTS)
        if suite != "core" and args.include_frequency_drops:
            variants.extend(SIMPLE_FREQUENCY_DROP_VARIANTS)
    if suite in {"exhaustive", "cxr-only"}:
        variants.extend(CXR_CORE_VARIANTS)
        if args.include_priors:
            variants.extend(CXR_PRIOR_VARIANTS)
        if args.include_frequency_drops:
            variants.extend(CXR_FREQUENCY_DROP_VARIANTS)
        variants.extend(CXR_TEXT_ADAPT_VARIANTS)

    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--plan",
        "screening",
        "--datasets",
        "qata",
        "--run-prefix",
        args.run_prefix,
        "--qata-epochs",
        str(args.epochs),
        "--qata-early-stop-patience",
        str(args.early_stop_patience),
        "--qata-batch-size",
        str(args.batch_size),
        "--qata-lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--save-last-every",
        str(args.save_last_every),
        "--qata-grad-clip-norm",
        str(args.grad_clip_norm),
        "--variants",
        *variants,
    ]
    cmd.append("--qata-use-amp" if args.use_amp else "--no-qata-use-amp")
    cmd.append("--continue-on-fail" if args.continue_on_fail else "--no-continue-on-fail")
    cmd.append("--resume-existing" if args.resume_existing else "--no-resume-existing")
    cmd.append("--skip-completed" if args.skip_completed else "--no-skip-completed")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--qata-data-root", args.qata_data_root])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    hf_home = ROOT / ".hf_cache"
    if hf_home.exists():
        env.setdefault("HF_HOME", str(hf_home))

    print(quote_cmd(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
