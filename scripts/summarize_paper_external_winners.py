from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PAPER_GROUPS = {
    "qata": {
        "Ours-best": "runs/qata_paper0516_qata_simple_native_zero_both_seed42/test_per_image_metrics.csv",
        "Ours-main-ResNet-CXR": "runs/qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42/test_per_image_metrics.csv",
        "FMISeg-official": "external_metrics/fmiseg_qata_official/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-our-prompt": "external_metrics/medclipsamv2_qata_our_prompt/test_per_image_metrics.csv",
    },
    "brain": {
        "Ours-structured-prompt": "runs/paper0623_brain_structured_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-our-prompt": "external_metrics/medclipsamv2_brain_our_prompt/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-native-prompt": "external_metrics/medclipsamv2_brain_native_prompt/test_per_image_metrics.csv",
    },
    "breast": {
        "Ours-structured-prompt": "runs/paper0623_breast_structured_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-our-prompt": "external_metrics/medclipsamv2_breast_our_prompt/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-native-prompt": "external_metrics/medclipsamv2_breast_native_prompt/test_per_image_metrics.csv",
    },
    "mosmed": {
        "Ours": "external_metrics/ours_mosmed/test_per_image_metrics.csv",
        "FMISeg": "external_metrics/fmiseg_mosmed/test_per_image_metrics.csv",
        "MedCLIP-SAMv2-our-prompt": "external_metrics/medclipsamv2_mosmed_our_prompt/test_per_image_metrics.csv",
    },
}


def read_rows(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        name_key = "mask_name" if "mask_name" in fields else "name"
        return {str(row[name_key]).strip(): row for row in reader if str(row.get(name_key, "")).strip()}


def f(row: dict, key: str) -> float:
    return float(row.get(key, "nan"))


def summarize(dataset: str, models: dict[str, str], out_dir: Path) -> tuple[list[str], list[dict]]:
    loaded = {}
    missing = []
    for model, rel in models.items():
        path = ROOT / rel
        if path.exists():
            loaded[model] = read_rows(path)
        else:
            missing.append((model, rel))

    lines = [f"## {dataset}", ""]
    if missing:
        lines.append("Missing baselines:")
        lines.extend(f"- `{model}`: `{rel}`" for model, rel in missing)
        lines.append("")
    if len(loaded) < 2:
        lines.append("Not enough per-image CSVs for winner comparison.")
        lines.append("")
        return lines, []

    common = sorted(set.intersection(*(set(v) for v in loaded.values())))
    lines.append(f"Compared samples: `{len(common)}`. Loaded models: `{len(loaded)}`.")
    lines.append("")

    dice_wins = defaultdict(int)
    iou_wins = defaultdict(int)
    dice_sum = defaultdict(float)
    iou_sum = defaultdict(float)
    cases = []
    for name in common:
        scores = {model: {"dice": f(rows[name], "dice"), "iou": f(rows[name], "iou")} for model, rows in loaded.items()}
        for model, score in scores.items():
            dice_sum[model] += score["dice"]
            iou_sum[model] += score["iou"]
        best_dice = max(score["dice"] for score in scores.values())
        best_iou = max(score["iou"] for score in scores.values())
        for model, score in scores.items():
            if abs(score["dice"] - best_dice) <= 1e-12:
                dice_wins[model] += 1
            if abs(score["iou"] - best_iou) <= 1e-12:
                iou_wins[model] += 1
        ranked = sorted(((m, s["dice"]) for m, s in scores.items()), key=lambda x: x[1], reverse=True)
        case = {
            "dataset": dataset,
            "mask_name": name,
            "best_model": ranked[0][0],
            "best_dice": ranked[0][1],
            "second_model": ranked[1][0] if len(ranked) > 1 else "",
            "second_dice": ranked[1][1] if len(ranked) > 1 else "",
            "dice_margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0,
        }
        for model, score in scores.items():
            safe = model.lower().replace(" ", "_").replace("-", "_")
            case[f"{safe}_dice"] = score["dice"]
            case[f"{safe}_iou"] = score["iou"]
        cases.append(case)

    n = max(len(common), 1)
    lines.append("| Model | Mean Dice | Mean IoU | Dice wins | IoU wins |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in sorted(loaded, key=lambda m: (-dice_sum[m] / n, m)):
        lines.append(f"| `{model}` | {dice_sum[model] / n:.6f} | {iou_sum[model] / n:.6f} | {dice_wins[model]} | {iou_wins[model]} |")
    lines.append("")
    lines.append("Top cases by Dice margin:")
    lines.append("| Case | Best model | Dice | Second | Margin |")
    lines.append("|---|---|---:|---|---:|")
    for row in sorted(cases, key=lambda x: x["dice_margin"], reverse=True)[:30]:
        lines.append(f"| `{row['mask_name']}` | `{row['best_model']}` | {row['best_dice']:.6f} | `{row['second_model']}` | {row['dice_margin']:.6f} |")
    lines.append("")

    csv_path = out_dir / f"{dataset}_paper_external_ranked_cases.csv"
    keys = sorted({key for row in cases for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sorted(cases, key=lambda x: x["dice_margin"], reverse=True))
    return lines, cases


def main() -> None:
    parser = argparse.ArgumentParser("Summarize paper external per-sample winners.")
    parser.add_argument("--output-dir", type=str, default="generated_figures/paper_external_winner_stats")
    args = parser.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Paper External Per-Sample Winner Statistics",
        "",
        "This table is for cross-model paper comparison only. Internal prompt/frequency ablations are intentionally excluded.",
        "",
    ]
    all_cases = []
    for dataset, models in PAPER_GROUPS.items():
        group_lines, cases = summarize(dataset, models, out_dir)
        lines.extend(group_lines)
        all_cases.extend(cases)

    (out_dir / "paper_external_winner_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "paper_external_winner_cases.json").write_text(json.dumps(all_cases, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'paper_external_winner_stats.md'}")
    print(f"Wrote {out_dir / 'paper_external_winner_cases.json'}")


if __name__ == "__main__":
    main()
