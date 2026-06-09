from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"

DEFAULT_VARIANTS = [
    "cxr_frozen_keep_both_empty",
    "cxr_frozen_keep_both_shuffle",
    "simple_native_keep_both",
    "simple_generic_keep_both",
    "cxr_frozen_learned_both",
    "cxr_lora8_keep_both",
]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def main() -> None:
    parser = argparse.ArgumentParser("Run the QATA diagnostic ablation batch sequentially.")
    parser.add_argument("--run-prefix", type=str, default="qata_diag0516")
    parser.add_argument("--qata-epochs", type=int, default=35)
    parser.add_argument("--qata-early-stop-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--include-unfreeze", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-zero-decoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--qata-data-root",
        type=str,
        default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2",
    )
    args = parser.parse_args()

    variants = list(DEFAULT_VARIANTS)
    if args.include_unfreeze:
        variants.append("cxr_unfreeze1_keep_both")
    if args.include_zero_decoder:
        variants.append("cxr_frozen_zero_decoder")

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
