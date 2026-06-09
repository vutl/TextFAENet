from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TEXT_SCRIPT = ROOT / "scripts" / "train_mosmed_text.py"
TRAIN_QATA_SCRIPT = ROOT / "scripts" / "train_qata.py"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def run_step(name: str, cmd: list[str], resume: bool) -> None:
    save_dir = Path(cmd[cmd.index("--save-dir") + 1])
    last_ckpt = save_dir / "last.pt"
    if resume and last_ckpt.exists() and "--resume-ckpt" not in cmd:
        cmd = [*cmd, "--resume-ckpt", str(last_ckpt)]

    print(f"\n=== {name} ===", flush=True)
    print(quote_cmd(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def no_text_cmd(
    data_root: str,
    dataset_format: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    merge_train_val: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_TEXT_SCRIPT),
        "--data-root",
        data_root,
        "--dataset-format",
        dataset_format,
        "--save-dir",
        save_dir,
        "--model-type",
        "faenet",
        "--no-text",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--image-size",
        "224",
        "--optimizer",
        "sgd",
        "--lr",
        "0.02",
        "--lr-scheduler",
        "poly",
        "--weight-decay",
        "1e-4",
        "--pos-weight",
        "auto",
        "--metric-thresholds",
        "0.5",
        "--save-last-every",
        "5",
    ]
    if merge_train_val:
        cmd.extend(["--merge-train-val", "--select-best-on", "train", "--early-stop-patience", "0", "--no-save-best"])
    else:
        cmd.extend(["--select-best-on", "val", "--early-stop-patience", "8"])
    return cmd


def qata_ablation_cmd(
    fusion_mode: str,
    data_root: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_QATA_SCRIPT),
        "--data-root",
        data_root,
        "--save-dir",
        save_dir,
        "--model-type",
        "lfaenet_tgfs_v2",
        "--fusion-mode",
        fusion_mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--image-size",
        "224",
        "--optimizer",
        "sgd",
        "--lr",
        "0.02",
        "--lr-scheduler",
        "poly",
        "--weight-decay",
        "1e-4",
        "--use-cxr-bert",
        "--cxr-bert-dir",
        "BiomedVLP-CXR-BERT-specialized",
        "--freeze-text-backbone",
        "--drop-hh-in-decoder",
        "--no-use-deep-supervision",
        "--save-last-every",
        "5",
        "--no-save-best-optimizer",
    ]


def main() -> None:
    parser = argparse.ArgumentParser("Run no-text baselines and QATA fusion ablations sequentially.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--qata-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-prefix", type=str, default="")
    parser.add_argument("--qata-data-root", type=str, default=r"D:\Documents\LMIS\FMISeg\data\QaTa-COV19-v2")
    args = parser.parse_args()

    prefix = f"{args.run_prefix}_" if args.run_prefix else ""
    steps: list[tuple[str, list[str]]] = [
        (
            "MosMed no-text FAENet",
            no_text_cmd(
                data_root=str(ROOT / "datasets" / "MosMed"),
                dataset_format="text_csv",
                save_dir=str(ROOT / "runs" / f"{prefix}mosmed_faenet_notext_e{args.epochs}"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                merge_train_val=False,
            ),
        ),
        (
            "Brain no-text FAENet train+val",
            no_text_cmd(
                data_root=str(ROOT / "datasets" / "brain_tumors"),
                dataset_format="prompt_folder",
                save_dir=str(ROOT / "runs" / f"{prefix}brain_tumors_faenet_notext_merge_trainbest_e{args.epochs}"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                merge_train_val=True,
            ),
        ),
        (
            "Breast no-text FAENet train+val",
            no_text_cmd(
                data_root=str(ROOT / "datasets" / "breast_tumors"),
                dataset_format="prompt_folder",
                save_dir=str(ROOT / "runs" / f"{prefix}breast_tumors_faenet_notext_merge_trainbest_e{args.epochs}"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                merge_train_val=True,
            ),
        ),
    ]

    for fusion_mode in ("encoder", "decoder", "both"):
        steps.append(
            (
                f"QATA LFAENet-TGFSv2 fusion={fusion_mode}",
                qata_ablation_cmd(
                    fusion_mode=fusion_mode,
                    data_root=args.qata_data_root,
                    save_dir=str(ROOT / "runs" / f"{prefix}qata_b4_e{args.epochs}_cxrbert_frozen_v2_{fusion_mode}"),
                    epochs=args.epochs,
                    batch_size=args.qata_batch_size,
                    num_workers=args.num_workers,
                ),
            )
        )

    for name, cmd in steps:
        run_step(name, cmd, resume=args.resume_existing)

    print("\nAll requested runs completed successfully.", flush=True)


if __name__ == "__main__":
    main()
