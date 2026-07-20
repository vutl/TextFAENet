from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LMIS_ROOT = ROOT.parent


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def derive_global_iou(global_dice: float | None) -> float | None:
    if global_dice is None or global_dice >= 2.0:
        return None
    return global_dice / (2.0 - global_dice)


def normalized_metrics(path: Path, kind: str = "standard") -> dict[str, Any]:
    data = read_json(path)
    empty = {
        "per_image_dice": None,
        "per_image_iou": None,
        "global_dice": None,
        "global_iou": None,
        "artifact": str(path),
    }
    if data is None:
        return empty
    if kind == "fmiseg":
        return {
            "per_image_dice": data.get("test_dice_image"),
            "per_image_iou": data.get("test_iou_image"),
            "global_dice": data.get("test_dice"),
            "global_iou": data.get("test_MIoU"),
            "artifact": str(path),
        }
    per_dice = data.get("per_image_dice", data.get("dice"))
    per_iou = data.get("per_image_iou", data.get("iou"))
    global_dice = data.get("global_dice")
    global_iou = data.get("global_iou")
    if global_iou is None:
        global_iou = derive_global_iou(global_dice)
    return {
        "per_image_dice": per_dice,
        "per_image_iou": per_iou,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "artifact": str(path),
    }


def run_metrics(run: str) -> dict[str, Any]:
    run_dir = ROOT / "runs" / run
    for filename in ("test_dual_metrics.json", "final_test.json"):
        path = run_dir / filename
        metrics = normalized_metrics(path)
        if any(metrics[key] is not None for key in ("per_image_dice", "global_dice")):
            return metrics
    return normalized_metrics(run_dir / "final_test.json")


def external_metrics(relative_path: str, kind: str = "standard") -> dict[str, Any]:
    return normalized_metrics(ROOT / relative_path, kind=kind)


def add_row(
    rows: list[dict[str, Any]],
    table: int,
    method: str,
    dataset_or_group: str,
    variant_or_prompt: str,
    metrics: dict[str, Any] | None = None,
    reported: tuple[float, float] | None = None,
    provenance: str = "local_full_test",
) -> None:
    metrics = metrics or {
        "per_image_dice": None,
        "per_image_iou": None,
        "global_dice": None,
        "global_iou": None,
        "artifact": None,
    }
    rows.append(
        {
            "table": table,
            "method": method,
            "dataset_or_group": dataset_or_group,
            "variant_or_prompt": variant_or_prompt,
            "reported_dice": None if reported is None else reported[0] / 100.0,
            "reported_iou": None if reported is None else reported[1] / 100.0,
            **metrics,
            "provenance": provenance,
        }
    )


def table2_rows(rows: list[dict[str, Any]]) -> None:
    reported = {
        "UNet": [(72.16, 60.51), (71.54, 62.34), (78.45, 68.76), (64.58, 50.73)],
        "nnUNet": [(77.76, 67.71), (73.77, 63.77), (80.42, 70.81), (72.59, 60.36)],
        "TransUNet": [(80.83, 71.52), (80.60, 71.68), (78.63, 69.13), (71.24, 58.44)],
        "VT-MFLV": [(84.63, 75.37), (78.05, 67.15), (83.34, 72.09), (75.61, 63.98)],
        "STPNet": [(79.66, 69.62), (71.25, 60.13), (80.63, 71.42), (76.18, 63.41)],
    }
    datasets = ["Brain MRI", "Breast US", "QaTa-COV19", "MosMedData+"]
    for method, values in reported.items():
        for dataset, pair in zip(datasets, values):
            add_row(
                rows,
                2,
                method,
                dataset,
                "source-reported baseline",
                reported=pair,
                provenance="source_reported_aggregation_not_recomputed",
            )

    medclip_paths = {
        "Brain MRI": "external_metrics/medclipsamv2_brain_structured/summary.json",
        "Breast US": "external_metrics/medclipsamv2_breast_structured/summary.json",
        "QaTa-COV19": "external_metrics/medclipsamv2_qata_structured/summary.json",
        "MosMedData+": "external_metrics/medclipsamv2_mosmed_structured/summary.json",
    }
    for dataset, path in medclip_paths.items():
        add_row(rows, 2, "MedCLIP-SAMv2", dataset, "our structured prompt", external_metrics(path))

    fmiseg_paths = {
        "Brain MRI": ("external_runs/fmiseg_brain_structured/final_test_metrics.json", "fmiseg"),
        "Breast US": (
            "../FMISeg/checkpoints/breast_tumors_text/final_test_metrics.json",
            "fmiseg",
        ),
        "QaTa-COV19": ("external_metrics/fmiseg_qata_official/summary.json", "standard"),
        "MosMedData+": ("external_runs/fmiseg_mosmed_structured/final_test_metrics.json", "fmiseg"),
    }
    for dataset, (path, kind) in fmiseg_paths.items():
        add_row(rows, 2, "FMISeg", dataset, "our structured prompt", external_metrics(path, kind))

    ours = {
        "Brain MRI": normalized_metrics(
            ROOT / "runs/paper0623_brain_structured_v3resnet50cxr_both_seed42/final_test.json"
        ),
        "Breast US": normalized_metrics(
            ROOT / "runs/paper0623_breast_structured_v3resnet50cxr_both_seed42/final_test.json"
        ),
        "QaTa-COV19": run_metrics(
            "qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42"
        ),
        "MosMedData+": normalized_metrics(ROOT / "runs/mosmed_v9e_448/final_test.json"),
    }
    for dataset, metrics in ours.items():
        add_row(rows, 2, "LFAENet-TGFS (ours)", dataset, "selected local full run", metrics)


def table3_rows(rows: list[dict[str, Any]]) -> None:
    for dataset, reported_pair, local_path in (
        (
            "Brain MRI",
            (80.03, 70.71),
            "external_metrics/medclipsamv2_brain_native/summary.json",
        ),
        (
            "Breast US",
            (78.87, 69.08),
            "external_metrics/medclipsamv2_breast_native/summary.json",
        ),
    ):
        add_row(
            rows,
            3,
            "MedCLIP-SAMv2",
            dataset,
            "original prompt (paper-reported)",
            reported=reported_pair,
            provenance="source_reported_aggregation_not_recomputed",
        )
        add_row(
            rows,
            3,
            "MedCLIP-SAMv2",
            dataset,
            "original prompt (local reproduction)",
            external_metrics(local_path),
        )
        add_row(
            rows,
            3,
            "MedCLIP-SAMv2",
            dataset,
            "our structured prompt",
            external_metrics(
                f"external_metrics/medclipsamv2_{'brain' if dataset.startswith('Brain') else 'breast'}_structured/summary.json"
            ),
        )

    for dataset, prompt, path in (
        (
            "Brain MRI",
            "original MedCLIP-SAMv2-style prompt",
            "external_runs/fmiseg_brain_medclip_prompt/final_test_metrics.json",
        ),
        (
            "Breast US",
            "original MedCLIP-SAMv2-style prompt",
            "external_runs/fmiseg_breast_medclip_prompt/final_test_metrics.json",
        ),
        (
            "Brain MRI",
            "our structured prompt",
            "external_runs/fmiseg_brain_structured/final_test_metrics.json",
        ),
        (
            "Breast US",
            "our structured prompt",
            "../FMISeg/checkpoints/breast_tumors_text/final_test_metrics.json",
        ),
    ):
        add_row(rows, 3, "FMISeg", dataset, prompt, external_metrics(path, "fmiseg"))

    for dataset, prompt, run in (
        ("Brain MRI", "original MedCLIP-SAMv2-style prompt", "paper0623_brain_medclip_prompt_v3resnet50cxr_both_seed42"),
        ("Breast US", "original MedCLIP-SAMv2-style prompt", "paper0623_breast_medclip_prompt_v3resnet50cxr_both_seed42"),
        ("Brain MRI", "our structured prompt", "paper0623_brain_structured_v3resnet50cxr_both_seed42"),
        ("Breast US", "our structured prompt", "paper0623_breast_structured_v3resnet50cxr_both_seed42"),
    ):
        add_row(rows, 3, "LFAENet-TGFS (ours)", dataset, prompt, run_metrics(run))


def table4_rows(rows: list[dict[str, Any]]) -> None:
    mappings = [
        ("A", "FAENet visual-only, scratch", "qata_paper0516_qata_faenet_visual_clean_seed42"),
        ("A", "FAENet visual-only, ResNet50", "qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42"),
        ("A", "TGFS decoder, ResNet50 + CXR-BERT", "qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42"),
        ("A", "TGFS decoder, ResNet50 + lightweight, learned HH", "qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42"),
        ("B", "Native, scratch lightweight, both", "qata_diag0516_qata_simple_native_keep_both_seed42"),
        ("B", "Empty, scratch lightweight, both", "qata_paper0516_qata_simple_empty_keep_both_seed42"),
        ("B", "Shuffled, scratch lightweight, both", "qata_paper0516_qata_simple_shuffle_keep_both_seed42"),
        ("B", "Generic, scratch lightweight, both", "qata_diag0516_qata_simple_generic_keep_both_seed42"),
        ("B", "Native, scratch CXR-BERT, both", "screening0506_qata_cxr_frozen_keep_both_seed42"),
        ("B", "Empty, scratch CXR-BERT, both", "qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42"),
        ("B", "Shuffled, scratch CXR-BERT, both", "qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42"),
        ("C", "Decoder-only, scratch lightweight", "qata_paper0516_qata_simple_native_keep_decoder_seed42"),
        ("C", "Encoder-decoder, scratch lightweight", "qata_diag0516_qata_simple_native_keep_both_seed42"),
        ("C", "Decoder-only, scratch CXR-BERT", "screening0506_qata_cxr_frozen_keep_decoder_seed42"),
        ("C", "Encoder-decoder, scratch CXR-BERT", "screening0506_qata_cxr_frozen_keep_both_seed42"),
    ]
    for group, variant, run in mappings:
        add_row(rows, 4, "QaTa ablation", group, variant, run_metrics(run))


def table5_rows(rows: list[dict[str, Any]], drop_run_prefix: str) -> None:
    old_keep_run = "qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42"
    new_keep_run = f"{drop_run_prefix}_qata_resnet50_simple_native_keep_decoder_seed42"
    new_keep_metrics = run_metrics(new_keep_run)
    keep_run = new_keep_run if new_keep_metrics["per_image_dice"] is not None else old_keep_run
    for block, variant, run in (
        ("A", "Keep HH", keep_run),
        ("A", "Zero HH", "qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42"),
        ("A", "Learned HH retention", "qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42"),
        ("B", "Full LL/LH/HL/HH", keep_run),
    ):
        add_row(rows, 5, "Frequency ablation", block, variant, run_metrics(run))

    old_runs = {
        "w/o LL": "qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42",
        "w/o LH": "qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42",
        "w/o HL": "qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42",
        "w/o HH": "qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42",
    }
    suffixes = {"w/o LL": "ll", "w/o LH": "lh", "w/o HL": "hl", "w/o HH": "hh"}
    for variant, old_run in old_runs.items():
        suffix = suffixes[variant]
        new_run = f"{drop_run_prefix}_qata_resnet50_simple_native_drop_{suffix}_decoder_seed42"
        new_metrics = run_metrics(new_run)
        metrics = new_metrics if new_metrics["per_image_dice"] is not None else run_metrics(old_run)
        add_row(rows, 5, "Frequency ablation", "B", variant, metrics)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Tables 2-5 dual-metric ledger",
        "",
        "All local values below are full-test results. `Reported` preserves the number copied from a source paper when prediction-level artifacts are unavailable; it is not automatically labelled per-image or global.",
        "",
    ]
    for table in range(2, 6):
        lines.extend(
            [
                f"## Table {table}",
                "",
                "| Method | Dataset/group | Variant/prompt | Reported Dice | Reported IoU | Per-image Dice | Per-image IoU | Global Dice | Global IoU | Provenance/artifact |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in (item for item in rows if item["table"] == table):
            provenance = row["provenance"]
            artifact = row.get("artifact")
            if artifact:
                provenance += f"; `{artifact}`"
            lines.append(
                "| {method} | {dataset} | {variant} | {rd} | {ri} | {pd} | {pi} | {gd} | {gi} | {prov} |".format(
                    method=row["method"],
                    dataset=row["dataset_or_group"],
                    variant=row["variant_or_prompt"],
                    rd=fmt(row["reported_dice"]),
                    ri=fmt(row["reported_iou"]),
                    pd=fmt(row["per_image_dice"]),
                    pi=fmt(row["per_image_iou"]),
                    gd=fmt(row["global_dice"]),
                    gi=fmt(row["global_iou"]),
                    prov=provenance,
                )
            )
        lines.append("")

    missing = [
        row
        for row in rows
        if row["reported_dice"] is None
        and any(row[key] is None for key in ("per_image_dice", "per_image_iou", "global_dice", "global_iou"))
    ]
    lines.extend(
        [
            "## Missing local dual metrics",
            "",
            "| Table | Method | Dataset/group | Variant/prompt |",
            "|---:|---|---|---|",
        ]
    )
    for row in missing:
        lines.append(
            f"| {row['table']} | {row['method']} | {row['dataset_or_group']} | {row['variant_or_prompt']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_runbook(results_path: Path, runbook_path: Path) -> None:
    if not runbook_path.is_file():
        return
    start_marker = "<!-- TABLES_2_5_RESULTS_START -->"
    end_marker = "<!-- TABLES_2_5_RESULTS_END -->"
    runbook = runbook_path.read_text(encoding="utf-8")
    if start_marker not in runbook or end_marker not in runbook:
        raise RuntimeError(f"Missing result markers in {runbook_path}")
    results = results_path.read_text(encoding="utf-8")
    table_start = results.find("## Table 2")
    if table_start < 0:
        raise RuntimeError(f"Cannot find Table 2 in {results_path}")
    embedded = results[table_start:].strip()
    prefix = runbook.split(start_marker, 1)[0]
    suffix = runbook.split(end_marker, 1)[1]
    updated = (
        prefix
        + start_marker
        + "\n\n"
        + embedded
        + "\n\n"
        + end_marker
        + suffix
    )
    runbook_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Collect all available Table 2-5 dual metrics.")
    parser.add_argument("--drop-run-prefix", default="paper_missing0715")
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "paper_tables_2_5_dual_metrics_20260715.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=ROOT / "paper_tables_2_5_dual_metrics_20260715.md"
    )
    parser.add_argument(
        "--runbook",
        type=Path,
        default=ROOT / "paper_missing_experiments_runbook_20260715.md",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    table2_rows(rows)
    table3_rows(rows)
    table4_rows(rows)
    table5_rows(rows, args.drop_run_prefix)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, args.output_md)
    update_runbook(args.output_md, args.runbook)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Updated {args.runbook}")


if __name__ == "__main__":
    main()
