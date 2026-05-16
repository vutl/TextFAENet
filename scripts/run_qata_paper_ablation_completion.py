from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"


MANDATORY_VARIANTS = [
    "simple_empty_keep_both",
    "simple_shuffle_keep_both",
    "simple_native_keep_decoder",
]

OPTIONAL_VARIANTS = [
    "simple_native_zero_both",
    "simple_native_learned_both",
]

BASELINE_VARIANTS = [
    "faenet_visual_clean",
]

INCOMPLETE_VARIANTS = [
    "cxr_lora8_keep_both",
]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def main() -> None:
    parser = argparse.ArgumentParser("Run the remaining QaTa paper ablations sequentially.")
    parser.add_argument("--run-prefix", type=str, default="qata_paper0516")
    parser.add_argument("--qata-epochs", type=int, default=35)
    parser.add_argument("--qata-early-stop-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--mandatory-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-optional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument(
        "--qata-data-root",
        type=str,
        default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2",
    )
    args = parser.parse_args()

    variants = list(MANDATORY_VARIANTS)
    if not args.mandatory_only:
        if args.include_optional:
            variants.extend(OPTIONAL_VARIANTS)
        if args.include_baseline:
            variants.extend(BASELINE_VARIANTS)
        if args.include_incomplete:
            variants.extend(INCOMPLETE_VARIANTS)

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
        str(args.qata_epochs),
        "--qata-early-stop-patience",
        str(args.qata_early_stop_patience),
        "--qata-batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--save-last-every",
        str(args.save_last_every),
        "--variants",
        *variants,
    ]
    cmd.append("--continue-on-fail" if args.continue_on_fail else "--no-continue-on-fail")
    cmd.append("--resume-existing" if args.resume_existing else "--no-resume-existing")
    cmd.append("--skip-completed" if args.skip_completed else "--no-skip-completed")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.batch_id is not None:
        cmd.extend(["--batch-id", args.batch_id])
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
