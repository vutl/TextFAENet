from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LMIS_ROOT = ROOT.parent
TRAIN_SCRIPT = ROOT / "scripts" / "train_brain_tumors.py"


def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    task_specs: list[dict[str, object]] = []
    datasets = set(args.datasets)
    prompt_protocols = set(args.prompt_protocols)

    if "brain" in datasets and "structured" in prompt_protocols:
        task_specs.append(
            {
                "name": "brain_structured",
                "data_root": ROOT / "datasets" / "brain_tumors",
                "csv_paths": None,
            }
        )
    if "breast" in datasets and "structured" in prompt_protocols:
        task_specs.append(
            {
                "name": "breast_structured",
                "data_root": ROOT / "datasets" / "breast_tumors",
                "csv_paths": None,
            }
        )
    if "brain" in datasets and "medclip" in prompt_protocols:
        data_root = LMIS_ROOT / "MedCLIP-SAMv2" / "data" / "brain_tumors"
        task_specs.append(
            {
                "name": "brain_medclip_prompt",
                "data_root": data_root,
                "csv_paths": {
                    "train": data_root / "FMISeg_train" / "train.csv",
                    "val": data_root / "FMISeg_val" / "val.csv",
                    "test": data_root / "FMISeg_test" / "test.csv",
                },
            }
        )
    if "breast" in datasets and "medclip" in prompt_protocols:
        data_root = LMIS_ROOT / "MedCLIP-SAMv2" / "data" / "breast_tumors"
        task_specs.append(
            {
                "name": "breast_medclip_prompt",
                "data_root": data_root,
                "csv_paths": {
                    "train": data_root / "FMISeg_train" / "train.csv",
                    "val": data_root / "FMISeg_val" / "val.csv",
                    "test": data_root / "FMISeg_test" / "test.csv",
                },
            }
        )

    return task_specs


def build_command(args: argparse.Namespace, task: dict[str, object]) -> list[str]:
    save_dir = ROOT / "runs" / f"{args.run_prefix}_{task['name']}_v3resnet50cxr_both_seed{args.seed}"
    cmd = [
        args.python,
        "-u",
        str(TRAIN_SCRIPT),
        "--data-root",
        str(task["data_root"]),
        "--save-dir",
        str(save_dir),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--save-last-every",
        str(args.save_last_every),
        "--preset-early-stop-patience",
        str(args.early_stop_patience),
    ]
    if args.resume_existing:
        cmd.append("--resume-existing")
    else:
        cmd.append("--no-resume-existing")

    csv_paths = task.get("csv_paths")
    if csv_paths:
        cmd.extend(
            [
                "--train-csv-path",
                str(csv_paths["train"]),
                "--val-csv-path",
                str(csv_paths["val"]),
                "--test-csv-path",
                str(csv_paths["test"]),
            ]
        )
    if args.smoke_test:
        cmd.append("--smoke-test")
    return cmd


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8", errors="replace") as log_f:
        log_f.write("\n" + "=" * 120 + "\n")
        log_f.write(" ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        "Run Brain/Breast prompt-protocol Text-FAENet v3 experiments sequentially."
    )
    parser.add_argument("--run-prefix", type=str, default="paper0623")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Patience passed through to train_brain_tumors.py after applying the v3 preset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["brain", "breast"],
        default=["brain", "breast"],
    )
    parser.add_argument(
        "--prompt-protocols",
        nargs="+",
        choices=["structured", "medclip"],
        default=["structured", "medclip"],
    )
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    tasks = build_tasks(args)
    if not tasks:
        raise SystemExit("No tasks selected.")

    log_dir = ROOT / "runs" / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{args.run_prefix}_brain_breast_prompt_protocol_{stamp}.log"

    failures: list[str] = []
    for task in tasks:
        save_dir = ROOT / "runs" / f"{args.run_prefix}_{task['name']}_v3resnet50cxr_both_seed{args.seed}"
        final_json = save_dir / "final_test.json"
        if args.skip_completed and final_json.exists():
            print(f"SKIP completed: {task['name']} ({final_json})")
            continue

        print(f"\n=== {task['name']} seed={args.seed} ===")
        cmd = build_command(args, task)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        rc = run_and_tee(cmd, log_path)
        if rc != 0:
            failures.append(f"{task['name']} rc={rc}")
            print(f"{task['name']} failed with exit code {rc}")
            if not args.continue_on_fail:
                break

    print(f"\nLogs saved to: {log_path}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Prompt-protocol suite completed.")


if __name__ == "__main__":
    main()
