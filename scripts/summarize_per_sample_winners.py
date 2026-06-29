from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_GROUPS = {
    "qata": {
        "Ours-best": "runs/qata_paper0516_qata_simple_native_zero_both_seed42/test_per_image_metrics.csv",
        "Ours-ResNet50-simple": "runs/qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42/test_per_image_metrics.csv",
        "Ours-ResNet50-CXR": "runs/qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42/test_per_image_metrics.csv",
        "Visual-only": "runs/qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42/test_per_image_metrics.csv",
        "Empty-prompt": "runs/qata_paper0516_qata_simple_empty_keep_both_seed42/test_per_image_metrics.csv",
        "Shuffle-prompt": "runs/qata_paper0516_qata_simple_shuffle_keep_both_seed42/test_per_image_metrics.csv",
    },
    "brain": {
        "Structured prompt": "runs/paper0623_brain_structured_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
        "MedCLIP prompt": "runs/paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
    },
    "breast": {
        "Structured prompt": "runs/paper0623_breast_structured_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
        "MedCLIP prompt": "runs/paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42/test_per_image_metrics.csv",
    },
    "mosmed": {},
}


def read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        name_key = "name" if "name" in (reader.fieldnames or []) else "mask_name"
        for row in reader:
            name = str(row.get(name_key, "")).strip()
            if not name:
                continue
            rows[name] = row
    return rows


def to_float(row: dict, key: str) -> float:
    return float(row.get(key, "nan"))


def gt_area_ratio(row: dict) -> float:
    if row.get("gt_area_ratio", "") != "":
        return float(row["gt_area_ratio"])
    target = row.get("target_sum", row.get("target_pixels", "0"))
    try:
        target_pixels = float(target)
    except ValueError:
        target_pixels = 0.0
    # Current exported masks are 224x224 for QaTa and 320x320 for brain/breast.
    # This value is only used for ranking/filtering, so infer the side length
    # from the common exported target scale when exact metadata is absent.
    denom = 224.0 * 224.0 if target_pixels <= 224.0 * 224.0 else 320.0 * 320.0
    return target_pixels / denom


def summarize_group(dataset: str, models: dict[str, str], out_dir: Path) -> tuple[list[str], list[dict]]:
    loaded = {}
    missing = []
    for model, rel in models.items():
        path = ROOT / rel
        if path.exists():
            loaded[model] = read_rows(path)
        else:
            missing.append(f"`{model}` missing `{rel}`")

    lines = []
    case_rows = []
    if not loaded:
        lines.append(f"## {dataset}\n")
        lines.append("No per-image metric CSVs available.\n")
        if missing:
            lines.extend(f"- {x}" for x in missing)
            lines.append("")
        return lines, case_rows

    common_names = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    dice_wins = defaultdict(int)
    iou_wins = defaultdict(int)
    strict_dice_wins = defaultdict(int)
    avg_dice = defaultdict(float)
    avg_iou = defaultdict(float)

    for name in common_names:
        per_model = {
            model: {
                "dice": to_float(rows[name], "dice"),
                "iou": to_float(rows[name], "iou"),
                "gt_area_ratio": gt_area_ratio(rows[name]),
                "text": rows[name].get("text", ""),
            }
            for model, rows in loaded.items()
        }
        for model, values in per_model.items():
            avg_dice[model] += values["dice"]
            avg_iou[model] += values["iou"]

        best_dice = max(v["dice"] for v in per_model.values())
        best_iou = max(v["iou"] for v in per_model.values())
        best_dice_models = [m for m, v in per_model.items() if abs(v["dice"] - best_dice) <= 1e-12]
        best_iou_models = [m for m, v in per_model.items() if abs(v["iou"] - best_iou) <= 1e-12]
        for model in best_dice_models:
            dice_wins[model] += 1
        for model in best_iou_models:
            iou_wins[model] += 1
        if len(best_dice_models) == 1:
            strict_dice_wins[best_dice_models[0]] += 1

        sorted_dice = sorted(((m, v["dice"]) for m, v in per_model.items()), key=lambda x: x[1], reverse=True)
        margin = sorted_dice[0][1] - sorted_dice[1][1] if len(sorted_dice) > 1 else 0.0
        case = {
            "dataset": dataset,
            "name": name,
            "best_model": sorted_dice[0][0],
            "best_dice": sorted_dice[0][1],
            "second_model": sorted_dice[1][0] if len(sorted_dice) > 1 else "",
            "second_dice": sorted_dice[1][1] if len(sorted_dice) > 1 else "",
            "dice_margin": margin,
            "gt_area_ratio": next(iter(per_model.values()))["gt_area_ratio"],
            "text": next((v["text"] for v in per_model.values() if v["text"]), ""),
        }
        for model, values in per_model.items():
            safe = model.lower().replace(" ", "_").replace("-", "_")
            case[f"{safe}_dice"] = values["dice"]
            case[f"{safe}_iou"] = values["iou"]
        case_rows.append(case)

    n = max(len(common_names), 1)
    lines.append(f"## {dataset}\n")
    lines.append(f"Compared samples: `{len(common_names)}`. Models with CSV: `{len(loaded)}`.\n")
    if missing:
        lines.append("Missing CSVs:")
        lines.extend(f"- {x}" for x in missing)
        lines.append("")
    lines.append("| Model | Mean Dice | Mean IoU | Dice wins | Strict Dice wins | IoU wins |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in sorted(loaded, key=lambda m: (-avg_dice[m] / n, m)):
        lines.append(
            f"| `{model}` | {avg_dice[model] / n:.6f} | {avg_iou[model] / n:.6f} | "
            f"{dice_wins[model]} | {strict_dice_wins[model]} | {iou_wins[model]} |"
        )
    lines.append("")

    top = sorted(case_rows, key=lambda r: r["dice_margin"], reverse=True)[:20]
    lines.append("Top cases by Dice margin:")
    lines.append("| Case | Best model | Dice | Second | Margin | GT area |")
    lines.append("|---|---|---:|---|---:|---:|")
    for row in top:
        lines.append(
            f"| `{row['name']}` | `{row['best_model']}` | {row['best_dice']:.6f} | "
            f"`{row['second_model']}` | {row['dice_margin']:.6f} | {row['gt_area_ratio']:.6f} |"
        )
    lines.append("")

    usable_cases = [r for r in case_rows if float(r["gt_area_ratio"]) > 0.001]
    top_usable = sorted(usable_cases, key=lambda r: r["dice_margin"], reverse=True)[:20]
    lines.append("Top usable non-empty cases by Dice margin (`gt_area_ratio > 0.001`):")
    lines.append("| Case | Best model | Dice | Second | Margin | GT area |")
    lines.append("|---|---|---:|---|---:|---:|")
    for row in top_usable:
        lines.append(
            f"| `{row['name']}` | `{row['best_model']}` | {row['best_dice']:.6f} | "
            f"`{row['second_model']}` | {row['dice_margin']:.6f} | {row['gt_area_ratio']:.6f} |"
        )
    lines.append("")

    lines.append("Best non-empty cases per winning model:")
    lines.append("| Winning model | Case | Dice | Second | Margin | GT area |")
    lines.append("|---|---|---:|---|---:|---:|")
    for model in sorted(loaded):
        model_cases = [r for r in usable_cases if r["best_model"] == model]
        for row in sorted(model_cases, key=lambda r: r["dice_margin"], reverse=True)[:5]:
            lines.append(
                f"| `{model}` | `{row['name']}` | {row['best_dice']:.6f} | "
                f"`{row['second_model']}` | {row['dice_margin']:.6f} | {row['gt_area_ratio']:.6f} |"
            )
    lines.append("")

    out_csv = out_dir / f"{dataset}_per_sample_ranked_cases.csv"
    all_keys = sorted({k for row in case_rows for k in row})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(sorted(case_rows, key=lambda r: r["dice_margin"], reverse=True))
    return lines, case_rows


def main() -> None:
    parser = argparse.ArgumentParser("Summarize per-sample winner counts across comparison models.")
    parser.add_argument("--output-dir", type=str, default="generated_figures/per_sample_winner_stats")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_lines = [
        "# Per-Sample Winner Statistics",
        "",
        "Winner = model with the highest per-image Dice/IoU for the same test sample.",
        "Ties are counted for all tied models; strict wins count only unique Dice winners.",
        "",
    ]
    all_cases = []
    for dataset, models in DEFAULT_GROUPS.items():
        lines, cases = summarize_group(dataset, models, out_dir)
        all_lines.extend(lines)
        all_cases.extend(cases)

    md_path = out_dir / "per_sample_winner_stats.md"
    md_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    json_path = out_dir / "per_sample_winner_cases.json"
    json_path.write_text(json.dumps(all_cases, indent=2), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
