from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MOSMED = ROOT / "scripts" / "train_mosmed_text.py"
TRAIN_QATA = ROOT / "scripts" / "train_qata.py"
QATA_ROOT = r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2"
MOSMED_ROOT = ROOT / "datasets" / "MosMed"
THRESHOLDS = "0.35,0.40,0.45,0.50,0.55"


VARIANTS: dict[str, dict[str, str | int | bool]] = {
    "simple_native_keep_decoder": {
        "use_cxr_bert": False,
        "prompt_mode": "native",
        "hh_drop_mode": "keep",
        "fusion_mode": "decoder",
    },
    "simple_generic_keep_decoder": {
        "use_cxr_bert": False,
        "prompt_mode": "generic",
        "hh_drop_mode": "keep",
        "fusion_mode": "decoder",
    },
    "cxr_frozen_zero_decoder": {
        "use_cxr_bert": True,
        "prompt_mode": "native",
        "hh_drop_mode": "zero",
        "fusion_mode": "decoder",
    },
    "cxr_frozen_keep_decoder": {
        "use_cxr_bert": True,
        "prompt_mode": "native",
        "hh_drop_mode": "keep",
        "fusion_mode": "decoder",
    },
    "cxr_frozen_keep_both": {
        "use_cxr_bert": True,
        "prompt_mode": "native",
        "hh_drop_mode": "keep",
        "fusion_mode": "both",
    },
    "cxr_lora8_keep_both": {
        "use_cxr_bert": True,
        "prompt_mode": "native",
        "hh_drop_mode": "keep",
        "fusion_mode": "both",
        "lora_r": 8,
    },
    "cxr_unfreeze1_keep_both": {
        "use_cxr_bert": True,
        "prompt_mode": "native",
        "hh_drop_mode": "keep",
        "fusion_mode": "both",
        "unfreeze_last_n": 1,
    },
}


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def add_variant_flags(cmd: list[str], variant: dict[str, str | int | bool]) -> list[str]:
    use_cxr_bert = bool(variant.get("use_cxr_bert", True))
    cmd.append("--use-cxr-bert" if use_cxr_bert else "--no-use-cxr-bert")
    cmd.extend(["--prompt-mode", str(variant.get("prompt_mode", "native"))])
    cmd.extend(["--hh-drop-mode", str(variant.get("hh_drop_mode", "keep"))])
    cmd.extend(["--fusion-mode", str(variant.get("fusion_mode", "decoder"))])
    cmd.extend(["--unfreeze-last-n", str(int(variant.get("unfreeze_last_n", 0)))])
    cmd.extend(["--lora-r", str(int(variant.get("lora_r", 0)))])
    cmd.append("--freeze-text-backbone")
    return cmd


def qata_cmd(args: argparse.Namespace, variant_name: str, variant: dict[str, str | int | bool], seed: int) -> list[str]:
    save_dir = ROOT / "runs" / f"{args.run_prefix}_qata_{variant_name}_seed{seed}"
    cmd = [
        sys.executable,
        str(TRAIN_QATA),
        "--data-root",
        args.qata_data_root,
        "--save-dir",
        str(save_dir),
        "--model-type",
        "lfaenet_tgfs_v2",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.qata_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--optimizer",
        "sgd",
        "--lr",
        str(args.qata_lr),
        "--lr-scheduler",
        "poly",
        "--weight-decay",
        "1e-4",
        "--seed",
        str(seed),
        "--metric-thresholds",
        THRESHOLDS,
        "--save-last-every",
        str(args.save_last_every),
        "--grad-clip-norm",
        str(args.qata_grad_clip_norm),
        "--no-save-best-optimizer",
    ]
    if not args.qata_deep_supervision:
        cmd.append("--no-use-deep-supervision")
    if not args.qata_use_amp:
        cmd.append("--no-use-amp")
    if args.save_debug_vis:
        cmd.extend(["--save-debug-vis", "--debug-vis-samples", str(args.debug_vis_samples)])
    return add_variant_flags(cmd, variant)


def mosmed_cmd(args: argparse.Namespace, variant_name: str, variant: dict[str, str | int | bool], seed: int) -> list[str]:
    save_dir = ROOT / "runs" / f"{args.run_prefix}_mosmed_{variant_name}_seed{seed}"
    cmd = [
        sys.executable,
        str(TRAIN_MOSMED),
        "--data-root",
        str(args.mosmed_data_root),
        "--dataset-format",
        "text_csv",
        "--save-dir",
        str(save_dir),
        "--model-type",
        "lfaenet_tgfs_v2",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.mosmed_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--optimizer",
        "adamw",
        "--lr",
        str(args.mosmed_lr),
        "--lr-scheduler",
        "cosine",
        "--weight-decay",
        "1e-4",
        "--seed",
        str(seed),
        "--pos-weight",
        "auto",
        "--metric-thresholds",
        THRESHOLDS,
        "--save-last-every",
        str(args.save_last_every),
        "--grad-clip-norm",
        str(args.mosmed_grad_clip_norm),
        "--select-best-on",
        "val",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--no-save-best-optimizer",
    ]
    if not args.mosmed_deep_supervision:
        cmd.append("--no-use-deep-supervision")
    if not args.mosmed_use_amp:
        cmd.append("--no-use-amp")
    if args.save_debug_vis:
        cmd.extend(["--save-debug-vis", "--debug-vis-samples", str(args.debug_vis_samples)])
    return add_variant_flags(cmd, variant)


def maybe_resume(cmd: list[str], resume: bool) -> list[str]:
    save_dir = Path(cmd[cmd.index("--save-dir") + 1])
    last_ckpt = save_dir / "last.pt"
    final_test = save_dir / "final_test.json"
    if final_test.exists() and not final_test_is_valid(final_test):
        return cmd
    if resume and last_ckpt.exists() and "--resume-ckpt" not in cmd:
        return [*cmd, "--resume-ckpt", str(last_ckpt)]
    return cmd


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def command_save_dir(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--save-dir") + 1])


def final_test_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    dice = float(data.get("dice", float("nan")))
    loss = float(data.get("loss", 0.0))
    return math.isfinite(dice) and math.isfinite(loss) and dice > 0.0


def run_command(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    log_dir: Path,
    dry_run: bool,
    skip_completed: bool,
) -> int:
    save_dir = command_save_dir(cmd)
    final_test = save_dir / "final_test.json"
    log_path = log_dir / f"{safe_name(name)}.log"

    if skip_completed and final_test_is_valid(final_test):
        message = f"SKIP completed: {name} ({final_test})"
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
        return 0
    if skip_completed and final_test.exists() and not final_test_is_valid(final_test):
        message = f"RERUN invalid final_test: {name} ({final_test})"
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    print(f"\n=== {name} ===", flush=True)
    print(quote_cmd(cmd), flush=True)
    with (log_dir / "commands.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n=== {name} ===\n{quote_cmd(cmd)}\n")

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
    parser = argparse.ArgumentParser("Run QATA/MosMed screening or confirmatory ablations.")
    parser.add_argument("--plan", choices=["screening", "confirmatory"], default="screening")
    parser.add_argument("--datasets", nargs="+", choices=["qata", "mosmed"], default=["qata", "mosmed"])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--run-prefix", type=str, default="screening")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--qata-batch-size", type=int, default=4)
    parser.add_argument("--mosmed-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--qata-lr", type=float, default=0.02)
    parser.add_argument("--mosmed-lr", type=float, default=3e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qata-use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mosmed-use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qata-deep-supervision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mosmed-deep-supervision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qata-grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--mosmed-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-debug-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-vis-samples", type=int, default=8)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--qata-data-root", type=str, default=QATA_ROOT)
    parser.add_argument("--mosmed-data-root", type=Path, default=MOSMED_ROOT)
    args = parser.parse_args()

    seeds = [42] if args.plan == "screening" else [42, 3407, 2026]
    env = os.environ.copy()
    hf_home = ROOT / ".hf_cache"
    if hf_home.exists():
        env.setdefault("HF_HOME", str(hf_home))

    batch_id = args.batch_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "runs" / "batch_logs" / f"{args.run_prefix}_{args.plan}_{batch_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    steps: list[tuple[str, list[str]]] = []
    for seed in seeds:
        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            if "qata" in args.datasets:
                steps.append((f"qata {variant_name} seed={seed}", qata_cmd(args, variant_name, variant, seed)))
            if "mosmed" in args.datasets:
                steps.append((f"mosmed {variant_name} seed={seed}", mosmed_cmd(args, variant_name, variant, seed)))

    for name, cmd in steps:
        cmd = maybe_resume(cmd, args.resume_existing)
        rc = run_command(
            name=name,
            cmd=cmd,
            env=env,
            log_dir=log_dir,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
        )
        if rc != 0:
            message = f"{name} failed with exit code {rc}"
            print(message, flush=True)
            with (log_dir / "failures.txt").open("a", encoding="utf-8") as f:
                f.write(message + "\n")
            if not args.continue_on_fail:
                raise SystemExit(message)

    print(f"\nLogs saved to: {log_dir}", flush=True)
    print("Ablation plan completed." if not args.dry_run else "Dry run completed.", flush=True)


if __name__ == "__main__":
    main()
