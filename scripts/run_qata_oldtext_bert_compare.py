from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_QATA = ROOT / "scripts" / "train_qata.py"
QATA_ROOT = r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2"
CXR_BERT = "BiomedVLP-CXR-BERT-specialized"
CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
THRESHOLDS = "0.35,0.40,0.45,0.50,0.55"


VARIANTS: dict[str, dict[str, str | float]] = {
    "v2old_cxrbert_decoder": {
        "model_type": "lfaenet_tgfs_v2_oldtext",
        "text_backbone": CXR_BERT,
        "lr": 0.02,
    },
    "v2old_clinicalbert_decoder": {
        "model_type": "lfaenet_tgfs_v2_oldtext",
        "text_backbone": CLINICAL_BERT,
        "lr": 0.02,
    },
    "v3old_resnet50_cxrbert_decoder": {
        "model_type": "lfaenet_tgfs_v3_oldtext",
        "text_backbone": CXR_BERT,
        "lr": 0.003,
    },
    "v3old_resnet50_clinicalbert_decoder": {
        "model_type": "lfaenet_tgfs_v3_oldtext",
        "text_backbone": CLINICAL_BERT,
        "lr": 0.003,
    },
}


def final_test_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("dice", 0.0)) > 0.0
    except Exception:
        return False


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def make_cmd(args: argparse.Namespace, name: str, variant: dict[str, str | float]) -> list[str]:
    save_dir = ROOT / "runs" / f"{args.run_prefix}_qata_{name}_seed{args.seed}"
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_QATA),
        "--data-root",
        args.qata_data_root,
        "--save-dir",
        str(save_dir),
        "--model-type",
        str(variant["model_type"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--optimizer",
        "sgd",
        "--lr",
        str(variant["lr"]),
        "--lr-scheduler",
        "poly",
        "--weight-decay",
        "1e-4",
        "--seed",
        str(args.seed),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--metric-thresholds",
        THRESHOLDS,
        "--save-last-every",
        str(args.save_last_every),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--no-save-best-optimizer",
        "--no-use-deep-supervision",
        "--no-use-amp",
        "--use-cxr-bert",
        "--cxr-bert-dir",
        str(variant["text_backbone"]),
        "--text-pooling",
        "mean",
        "--prompt-mode",
        "native",
        "--hh-drop-mode",
        "keep",
        "--fusion-mode",
        "decoder",
        "--freeze-text-backbone",
        "--unfreeze-last-n",
        "0",
        "--lora-r",
        "0",
        "--text-dim",
        "256",
        "--max-text-len",
        "64",
    ]
    if str(variant["model_type"]) == "lfaenet_tgfs_v3_oldtext":
        cmd.extend([
            "--v3-encoder-type",
            "resnet50",
            "--pretrained-image-encoder",
            "--freeze-encoder-bn",
            "--encoder-text-fusion",
            "film",
        ])
    if args.resume_existing:
        last_ckpt = save_dir / "last.pt"
        if last_ckpt.exists():
            cmd.extend(["--resume-ckpt", str(last_ckpt)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser("Run old-text-logic QaTa BERT comparison.")
    parser.add_argument("--run-prefix", type=str, default="qata_oldtext_bert0629")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qata-data-root", type=str, default=QATA_ROOT)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

    log_dir = ROOT / "runs" / "batch_logs" / f"{args.run_prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for name in args.variants:
        cmd = make_cmd(args, name, VARIANTS[name])
        save_dir = Path(cmd[cmd.index("--save-dir") + 1])
        if args.skip_completed and final_test_is_valid(save_dir / "final_test.json"):
            print(f"SKIP completed: {name} ({save_dir})", flush=True)
            continue
        print(f"\n=== qata {name} seed={args.seed} ===", flush=True)
        print(quote_cmd(cmd), flush=True)
        with (log_dir / "commands.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n=== qata {name} seed={args.seed} ===\n{quote_cmd(cmd)}\n")
        if args.dry_run:
            continue
        with (log_dir / f"{name}.log").open("a", encoding="utf-8") as f:
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
                f.write(line)
            rc = proc.wait()
        if rc != 0:
            msg = f"qata {name} failed with exit code {rc}"
            print(msg, flush=True)
            with (log_dir / "failures.txt").open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            if not args.continue_on_fail:
                raise SystemExit(rc)

    print(f"\nLogs saved to: {log_dir}", flush=True)
    print("Old-text BERT comparison completed." if not args.dry_run else "Dry run completed.", flush=True)


if __name__ == "__main__":
    main()
