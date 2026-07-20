from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LMIS_ROOT = ROOT.parent
FMISEG_ROOT = LMIS_ROOT / "FMISeg"
MEDCLIP_ROOT = LMIS_ROOT / "MedCLIP-SAMv2"
QATA_ROOT = FMISEG_ROOT / "data" / "QaTa-COV19-v2"

TASK_CHOICES = (
    "medclip_structured",
    "fmiseg_structured",
    "fmiseg_prompt_transfer",
    "qata_drop_bands",
    "qata_main_full",
)


class TeeLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_command(
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log: TeeLog,
    dry_run: bool,
) -> int:
    printable = subprocess.list2cmdline(cmd)
    log.write(f"\n=== {name} ===")
    log.write(f"cwd={cwd}")
    log.write(printable)
    if dry_run:
        return 0
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line.rstrip("\r\n"))
    return process.wait()


def medclip_commands(python: str) -> list[tuple[str, list[str], Path, Path]]:
    generic = ROOT / "scripts" / "run_medclipsamv2_prompted_folder_zeroshot.py"
    qata = ROOT / "scripts" / "run_medclipsamv2_qata_zeroshot.py"
    return [
        (
            "MedCLIP-SAMv2 Brain structured prompts",
            [
                python,
                "-u",
                str(generic),
                "--dataset-name",
                "brain_tumors",
                "--dataset-root",
                str(ROOT / "datasets" / "brain_tumors"),
                "--prompt-csv",
                str(ROOT / "datasets" / "brain_tumors" / "test.csv"),
                "--out-dir",
                str(ROOT / "external_metrics" / "medclipsamv2_brain_structured"),
                "--medclip-root",
                str(MEDCLIP_ROOT),
                "--device",
                "cuda",
                "--num-contours",
                "1",
            ],
            ROOT,
            ROOT / "external_metrics" / "medclipsamv2_brain_structured" / "summary.json",
        ),
        (
            "MedCLIP-SAMv2 Breast structured prompts",
            [
                python,
                "-u",
                str(generic),
                "--dataset-name",
                "breast_tumors",
                "--dataset-root",
                str(ROOT / "datasets" / "breast_tumors"),
                "--prompt-csv",
                str(ROOT / "datasets" / "breast_tumors" / "test.csv"),
                "--out-dir",
                str(ROOT / "external_metrics" / "medclipsamv2_breast_structured"),
                "--medclip-root",
                str(MEDCLIP_ROOT),
                "--device",
                "cuda",
                "--num-contours",
                "1",
            ],
            ROOT,
            ROOT / "external_metrics" / "medclipsamv2_breast_structured" / "summary.json",
        ),
        (
            "MedCLIP-SAMv2 QaTa structured prompts",
            [
                python,
                "-u",
                str(qata),
                "--medclip-root",
                str(MEDCLIP_ROOT),
                "--qata-root",
                str(QATA_ROOT),
                "--split",
                "test",
                "--out-dir",
                str(ROOT / "external_metrics" / "medclipsamv2_qata_structured"),
                "--device",
                "cuda",
                "--num-contours",
                "2",
            ],
            ROOT,
            ROOT / "external_metrics" / "medclipsamv2_qata_structured" / "summary.json",
        ),
        (
            "MedCLIP-SAMv2 MosMed structured prompts",
            [
                python,
                "-u",
                str(generic),
                "--dataset-name",
                "mosmed",
                "--dataset-root",
                str(ROOT / "datasets" / "MosMed"),
                "--prompt-csv",
                str(ROOT / "Test_text_MosMedData+(in).csv"),
                "--image-subdir",
                "frames",
                "--mask-subdir",
                "masks",
                "--out-dir",
                str(ROOT / "external_metrics" / "medclipsamv2_mosmed_structured"),
                "--medclip-root",
                str(MEDCLIP_ROOT),
                "--device",
                "cuda",
                "--num-contours",
                "2",
            ],
            ROOT,
            ROOT / "external_metrics" / "medclipsamv2_mosmed_structured" / "summary.json",
        ),
    ]


def fmiseg_commands(python: str, task: str) -> list[tuple[str, list[str], Path, Path]]:
    if task == "fmiseg_structured":
        configs = [
            ("FMISeg Brain structured prompts", "fmiseg_brain_structured"),
            ("FMISeg MosMed structured prompts", "fmiseg_mosmed_structured"),
        ]
    else:
        configs = [
            ("FMISeg Brain MedCLIP-style prompts", "fmiseg_brain_medclip_prompt"),
            ("FMISeg Breast MedCLIP-style prompts", "fmiseg_breast_medclip_prompt"),
        ]
    commands: list[tuple[str, list[str], Path, Path]] = []
    if task == "fmiseg_prompt_transfer":
        commands.append(
            (
                "Prepare path-explicit FMISeg MedCLIP-style prompt CSVs",
                [
                    python,
                    "-u",
                    str(ROOT / "scripts" / "prepare_fmiseg_medclip_prompt_csvs.py"),
                ],
                ROOT,
                ROOT
                / "external_runs"
                / "fmiseg_prompt_csvs"
                / "breast_tumors"
                / "test.csv",
            )
        )
    for label, stem in configs:
        config = ROOT / "configs" / "paper_missing" / f"{stem}.yaml"
        result = ROOT / "external_runs" / stem / "final_test_metrics.json"
        commands.append(
            (
                label,
                [python, "-u", str(FMISEG_ROOT / "train.py"), "--config", str(config)],
                FMISEG_ROOT,
                result,
            )
        )
    return commands


def qata_drop_commands(
    python: str, run_prefix: str, epochs: int, patience: int, batch_size: int, workers: int
) -> list[tuple[str, list[str], Path, Path]]:
    variants = [
        "resnet50_simple_native_drop_ll_decoder",
        "resnet50_simple_native_drop_lh_decoder",
        "resnet50_simple_native_drop_hl_decoder",
        "resnet50_simple_native_drop_hh_decoder",
    ]
    train_cmd = [
        python,
        "-u",
        str(ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"),
        "--plan",
        "screening",
        "--datasets",
        "qata",
        "--run-prefix",
        run_prefix,
        "--qata-epochs",
        str(epochs),
        "--qata-early-stop-patience",
        str(patience),
        "--qata-batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--save-last-every",
        "5",
        "--variants",
        *variants,
        "--continue-on-fail",
        "--resume-existing",
        "--skip-completed",
        "--qata-data-root",
        str(QATA_ROOT),
    ]
    run_dirs = [ROOT / "runs" / f"{run_prefix}_qata_{variant}_seed42" for variant in variants]
    eval_cmd = [
        python,
        "-u",
        str(ROOT / "scripts" / "evaluate_qata_dual_metrics.py"),
        "--batch-size",
        "8",
        "--num-workers",
        str(workers),
        "--no-use-amp",
        "--continue-on-fail",
        "--output-json",
        str(ROOT / f"{run_prefix}_drop_band_dual_metrics.json"),
        "--output-md",
        str(ROOT / f"{run_prefix}_drop_band_dual_metrics.md"),
        "--runs",
        *[str(path) for path in run_dirs],
    ]
    return [
        (
            "QaTa four drop-band retrains",
            train_cmd,
            ROOT,
            ROOT / "runs" / f".{run_prefix}_always_check_all_drop_runs",
        ),
        (
            "QaTa drop-band dual-metric evaluation",
            eval_cmd,
            ROOT,
            ROOT / f".{run_prefix}_always_refresh_drop_metrics",
        ),
    ]


def qata_main_full_commands(
    python: str, run_prefix: str, epochs: int, patience: int, batch_size: int, workers: int
) -> list[tuple[str, list[str], Path, Path]]:
    variant = "resnet50_simple_native_keep_decoder"
    run_dir = ROOT / "runs" / f"{run_prefix}_qata_{variant}_seed42"
    train_cmd = [
        python,
        "-u",
        str(ROOT / "scripts" / "run_qata_mosmed_ablation_plan.py"),
        "--plan",
        "screening",
        "--datasets",
        "qata",
        "--run-prefix",
        run_prefix,
        "--qata-epochs",
        str(epochs),
        "--qata-early-stop-patience",
        str(patience),
        "--qata-batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--save-last-every",
        "5",
        "--variants",
        variant,
        "--continue-on-fail",
        "--resume-existing",
        "--skip-completed",
        "--qata-data-root",
        str(QATA_ROOT),
    ]
    output_json = ROOT / f"{run_prefix}_main_full_dual_metrics.json"
    eval_cmd = [
        python,
        "-u",
        str(ROOT / "scripts" / "evaluate_qata_dual_metrics.py"),
        "--batch-size",
        "8",
        "--num-workers",
        str(workers),
        "--no-use-amp",
        "--continue-on-fail",
        "--output-json",
        str(output_json),
        "--output-md",
        str(ROOT / f"{run_prefix}_main_full_dual_metrics.md"),
        "--runs",
        str(run_dir),
    ]
    return [
        (
            "QaTa full-band main reference retrain",
            train_cmd,
            ROOT,
            ROOT / f".{run_prefix}_always_check_main_full",
        ),
        (
            "QaTa full-band main reference dual-metric evaluation",
            eval_cmd,
            ROOT,
            ROOT / f".{run_prefix}_always_refresh_main_full_metrics",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        "Run all missing paper experiments sequentially with resume and skip support."
    )
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASK_CHOICES))
    parser.add_argument("--run-prefix", default="paper_missing0715")
    parser.add_argument("--qata-epochs", type=int, default=35)
    parser.add_argument("--qata-early-stop-patience", type=int, default=8)
    parser.add_argument("--qata-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "runs" / "batch_logs" / f"{args.run_prefix}_{stamp}.log"
    log = TeeLog(log_path)
    failures: list[dict[str, object]] = []
    try:
        commands: list[tuple[str, list[str], Path, Path]] = []
        for task in args.tasks:
            if task == "medclip_structured":
                commands.extend(medclip_commands(python))
            elif task in {"fmiseg_structured", "fmiseg_prompt_transfer"}:
                commands.extend(fmiseg_commands(python, task))
            elif task == "qata_drop_bands":
                commands.extend(
                    qata_drop_commands(
                        python,
                        args.run_prefix,
                        args.qata_epochs,
                        args.qata_early_stop_patience,
                        args.qata_batch_size,
                        args.num_workers,
                    )
                )
            elif task == "qata_main_full":
                commands.extend(
                    qata_main_full_commands(
                        python,
                        args.run_prefix,
                        args.qata_epochs,
                        args.qata_early_stop_patience,
                        args.qata_batch_size,
                        args.num_workers,
                    )
                )

        for name, cmd, cwd, completion_file in commands:
            if args.skip_completed and completion_file.is_file():
                log.write(f"SKIP completed: {name} ({completion_file})")
                continue
            code = run_command(name, cmd, cwd, env, log, args.dry_run)
            if code != 0:
                failure = {"task": name, "exit_code": code, "command": cmd}
                failures.append(failure)
                log.write(f"FAILED: {name} exited with {code}")
                if not args.continue_on_fail:
                    break

        collector = ROOT / "scripts" / "collect_paper_tables_2_5_dual_metrics.py"
        if collector.is_file() and not args.dry_run:
            run_command(
                "Refresh Table 2-5 dual-metric ledger",
                [python, "-u", str(collector), "--drop-run-prefix", args.run_prefix],
                ROOT,
                env,
                log,
                dry_run=False,
            )
    finally:
        failure_path = ROOT / "runs" / "batch_logs" / f"{args.run_prefix}_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        log.write(f"\nLog: {log_path}")
        log.write(f"Failures: {len(failures)} ({failure_path})")
        log.close()

    if failures and not args.continue_on_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
