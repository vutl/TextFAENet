from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SOURCE_FILES = [
    ROOT / "brain_breast_prompt_protocol_results_20260626.md",
    ROOT / "qata_dual_metrics_resnet_cxr_vs_best_ablation.md",
    ROOT / "qata_results_summary_20260530.md",
    ROOT / "qata_dual_metrics_resnet_cxr_vs_best_ablation.json",
]


def extract_run_names() -> list[str]:
    names: list[str] = []

    for source in SOURCE_FILES:
        if not source.exists():
            continue
        if source.suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(data, list):
                candidates = [row.get("run") for row in data if isinstance(row, dict)]
            else:
                candidates = []
        else:
            text = source.read_text(encoding="utf-8")
            candidates = re.findall(r"`([^`]+)`", text)

        for raw in candidates:
            if not raw:
                continue
            item = str(raw).replace("\\", "/")
            if "*" in item:
                continue
            if item.startswith("runs/"):
                item = item.split("/", 1)[1].split("/", 1)[0]
            if "/" in item:
                continue
            if not (
                item.startswith("paper0623_")
                or item.startswith("qata_")
                or item.startswith("screening0506_")
            ):
                continue
            if item not in names:
                names.append(item)
    return names


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def metric_from_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    dice_values: list[float] = []
    iou_values: list[float] = []
    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0
    total_union = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dice_values.append(float(row["dice"]))
            iou_values.append(float(row["iou"]))
            inter = float(row["intersection"])
            pred = float(row["pred_pixels"])
            target = float(row["target_pixels"])
            total_intersection += inter
            total_pred += pred
            total_target += target
            total_union += pred + target - inter
    if not dice_values:
        return None
    eps = 1e-6
    return {
        "per_image_dice": sum(dice_values) / len(dice_values),
        "per_image_iou": sum(iou_values) / len(iou_values),
        "global_dice": (2.0 * total_intersection + eps) / (total_pred + total_target + eps),
        "global_iou": (total_intersection + eps) / (total_union + eps),
        "num_images": len(dice_values),
        "source": "test_per_image_metrics.csv",
    }


def normalize_final_metrics(final: dict[str, Any]) -> dict[str, Any]:
    per_dice = final.get("per_image_dice", final.get("dice"))
    per_iou = final.get("per_image_iou", final.get("iou"))
    global_dice = final.get("global_dice")
    global_iou = final.get("global_iou")
    return {
        "per_image_dice": per_dice,
        "per_image_iou": per_iou,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "num_images": final.get("num_test_images", final.get("num_images")),
        "best_epoch": final.get("best_epoch"),
        "best_threshold": final.get("best_threshold"),
        "source": "final_test.json" if global_dice is not None else "final_test.json_per_image_only",
    }


def collect_one(run_name: str) -> dict[str, Any]:
    run_dir = RUNS / run_name
    row: dict[str, Any] = {
        "run": run_name,
        "exists": run_dir.exists(),
        "status": "missing_dir",
        "per_image_dice": None,
        "per_image_iou": None,
        "global_dice": None,
        "global_iou": None,
        "num_images": None,
        "best_epoch": None,
        "best_threshold": None,
        "source": "-",
        "can_rerun_from_checkpoint": False,
        "checkpoint_files": "",
    }
    if not run_dir.exists():
        return row

    ckpts = [name for name in ("best.pt", "last.pt") if (run_dir / name).exists()]
    row["checkpoint_files"] = ",".join(ckpts)
    row["can_rerun_from_checkpoint"] = bool(ckpts and (run_dir / "config.json").exists())

    dual = read_json(run_dir / "test_dual_metrics.json")
    if dual is not None:
        row.update(
            {
                "status": "dual_available",
                "per_image_dice": dual.get("per_image_dice"),
                "per_image_iou": dual.get("per_image_iou"),
                "global_dice": dual.get("global_dice"),
                "global_iou": dual.get("global_iou"),
                "num_images": dual.get("num_images"),
                "best_epoch": dual.get("checkpoint_epoch"),
                "best_threshold": dual.get("threshold"),
                "source": "test_dual_metrics.json",
            }
        )
        return row

    csv_metrics = metric_from_csv(run_dir / "test_per_image_metrics.csv")
    if csv_metrics is not None:
        row.update({"status": "dual_available", **csv_metrics})
        final = read_json(run_dir / "final_test.json")
        if final:
            row["best_epoch"] = final.get("best_epoch")
            row["best_threshold"] = final.get("best_threshold")
        return row

    final = read_json(run_dir / "final_test.json")
    if final is not None:
        metrics = normalize_final_metrics(final)
        row.update(metrics)
        row["status"] = "dual_available" if metrics["global_dice"] is not None else "per_image_only"
        return row

    row["status"] = "no_final"
    return row


def fmt_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.2f}"
    except Exception:
        return "-"


def fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    dual_count = sum(1 for r in rows if r["global_dice"] is not None and r["global_iou"] is not None)
    per_only_count = sum(1 for r in rows if r["per_image_dice"] is not None and r["global_dice"] is None)
    no_final_count = sum(1 for r in rows if r["per_image_dice"] is None)
    rerunnable_count = sum(
        1
        for r in rows
        if r["global_dice"] is None and r["can_rerun_from_checkpoint"]
    )

    lines = [
        "# Available Per-Image and Global Metrics - 2026-06-28",
        "",
        "This file aggregates runs referenced by:",
        "",
        "- `brain_breast_prompt_protocol_results_20260626.md`",
        "- `qata_dual_metrics_resnet_cxr_vs_best_ablation.md`",
        "- `qata_results_summary_20260530.md`",
        "- `qata_dual_metrics_resnet_cxr_vs_best_ablation.json`",
        "",
        "Rules:",
        "",
        "- If `test_dual_metrics.json` exists, use it.",
        "- Else if `test_per_image_metrics.csv` exists, recompute per-image and global metrics from stored intersections/pixel counts.",
        "- Else if `final_test.json` contains `global_*`, use it.",
        "- Else use `final_test.json` Dice/IoU as per-image only and leave global as `-`.",
        "- If a checkpoint remains, `can_rerun_from_checkpoint=yes` means global metrics can be recovered by rerunning inference.",
        "",
        f"Summary: `{dual_count}` runs have both per-image and global metrics now; `{per_only_count}` have per-image only; `{no_final_count}` have no final test metrics; `{rerunnable_count}` can be rerun from a checkpoint to recover global metrics.",
        "",
        "## Metrics Table",
        "",
        "| Run | Status | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Best epoch | Thr | Images | Source | Checkpoint rerun? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        rerun = "yes" if r["global_dice"] is None and r["can_rerun_from_checkpoint"] else "no"
        lines.append(
            f"| `{r['run']}` | {r['status']} | {fmt_percent(r['per_image_dice'])} | {fmt_percent(r['per_image_iou'])} | "
            f"{fmt_percent(r['global_dice'])} | {fmt_percent(r['global_iou'])} | {fmt_value(r['best_epoch'])} | "
            f"{fmt_value(r['best_threshold'])} | {fmt_value(r['num_images'])} | `{r['source']}` | {rerun} |"
        )

    lines.extend(
        [
            "",
            "## Runs Needing Inference Rerun For Global Metrics",
            "",
            "These runs have old per-image-only `final_test.json` metrics but still have enough checkpoint/config files to recompute global Dice/IoU.",
            "",
        ]
    )
    rerunnable = [r for r in rows if r["global_dice"] is None and r["can_rerun_from_checkpoint"]]
    if rerunnable:
        for r in rerunnable:
            lines.append(f"- `runs/{r['run']}` ({r['checkpoint_files']})")
        lines.extend(
            [
                "",
                "Command to recover dual metrics for all rerunnable QaTa runs:",
                "",
                "```powershell",
                "$env:HF_HOME=(Resolve-Path .hf_cache).Path",
                "$env:TORCH_HOME=(Resolve-Path .torch_cache).Path",
                "D:\\anaconda3\\python.exe -u scripts\\evaluate_qata_dual_metrics.py `",
                "  --batch-size 8 --num-workers 4 --no-use-amp `",
                "  --output-json qata_dual_metrics_recovered_20260628.json `",
                "  --output-md qata_dual_metrics_recovered_20260628.md `",
                "  --runs `",
            ]
        )
        for idx, r in enumerate(rerunnable):
            suffix = " `" if idx < len(rerunnable) - 1 else ""
            lines.append(f"  runs/{r['run']}{suffix}")
        lines.append("```")
        lines.extend(
            [
                "",
                "This command will take a while because it reruns test inference. It should be run only when you actually need global metrics for every old QaTa row.",
            ]
        )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Runs That Cannot Be Recovered Without Retraining Or A Checkpoint",
            "",
        ]
    )
    blocked = [r for r in rows if r["global_dice"] is None and not r["can_rerun_from_checkpoint"]]
    if blocked:
        for r in blocked:
            lines.append(f"- `{r['run']}`: status `{r['status']}`, source `{r['source']}`")
    else:
        lines.append("- None.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    run_names = extract_run_names()
    rows = [collect_one(name) for name in run_names]
    output_json = ROOT / "available_dual_metrics_20260628.json"
    output_md = ROOT / "available_dual_metrics_20260628.md"
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, output_md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
