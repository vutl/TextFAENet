from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_missing_runs(metrics_path: Path) -> list[str]:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs: list[str] = []
    for row in rows:
        if row.get("global_dice") is None and row.get("can_rerun_from_checkpoint"):
            runs.append(f"runs/{row['run']}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser("Run QaTa dual-metric inference for missing rerunnable runs.")
    parser.add_argument("--metrics-json", type=str, default="available_dual_metrics_20260628.json")
    parser.add_argument("--output-json", type=str, default="qata_dual_metrics_recovered_20260628.json")
    parser.add_argument("--output-md", type=str, default="qata_dual_metrics_recovered_20260628.md")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    metrics_path = ROOT / args.metrics_json
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics json: {metrics_path}")

    runs = load_missing_runs(metrics_path)
    print(f"Missing rerunnable runs: {len(runs)}", flush=True)
    for run in runs:
        print(run, flush=True)
    if not runs:
        return

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "evaluate_qata_dual_metrics.py"),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--continue-on-fail",
        "--output-json",
        args.output_json,
        "--output-md",
        args.output_md,
    ]
    cmd.append("--use-amp" if args.use_amp else "--no-use-amp")
    cmd.append("--runs")
    cmd.extend(runs)

    print("\nCommand:", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
