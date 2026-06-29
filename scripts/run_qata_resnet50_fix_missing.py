from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"

MISSING_OR_FAILED = [
    "resnet50_faenet_visual_clean",
    "resnet50_simple_native_keep_decoder",
    "resnet50_simple_empty_keep_decoder",
    "resnet50_simple_native_drop_ll_decoder",
    "resnet50_simple_native_drop_hh_decoder",
]


def main() -> None:
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--plan",
        "screening",
        "--datasets",
        "qata",
        "--run-prefix",
        "qata_resnet0523_fp32fix",
        "--qata-epochs",
        "50",
        "--qata-early-stop-patience",
        "10",
        "--qata-batch-size",
        "4",
        "--qata-lr",
        "0.003",
        "--num-workers",
        "4",
        "--save-last-every",
        "5",
        "--qata-grad-clip-norm",
        "1.0",
        "--variants",
        *MISSING_OR_FAILED,
        "--no-qata-use-amp",
        "--continue-on-fail",
        "--no-resume-existing",
        "--skip-completed",
        "--qata-data-root",
        r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2",
    ]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
