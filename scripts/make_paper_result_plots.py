from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "generated_figures" / "paper_results_20260624"


def load_metric(run_name: str) -> dict[str, float | int | str | None]:
    final_path = RUNS / run_name / "final_test.json"
    if not final_path.exists():
        return {
            "run": run_name,
            "dice": None,
            "iou": None,
            "epoch": None,
            "threshold": None,
        }
    data = json.loads(final_path.read_text(encoding="utf-8"))
    dice = data.get("per_image_dice", data.get("dice"))
    iou = data.get("per_image_iou", data.get("iou"))
    return {
        "run": run_name,
        "dice": float(dice) if dice is not None else None,
        "iou": float(iou) if iou is not None else None,
        "epoch": data.get("best_epoch"),
        "threshold": data.get("best_threshold"),
    }


def percent(x: float | None) -> float:
    if x is None:
        return float("nan")
    return 100.0 * float(x)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf7",
        }
    )


def annotate_bars(ax, bars, values, dy: float = 0.35) -> None:
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_main_qata_comparison() -> list[dict[str, object]]:
    specs = [
        (
            "Scratch FAENet",
            "qata_paper0516_qata_faenet_visual_clean_seed42",
            "visual-only",
        ),
        (
            "ResNet50 FAENet",
            "qata_resnet0523_fp32fix_qata_resnet50_faenet_visual_clean_seed42",
            "visual-only",
        ),
        (
            "ResNet50 + CXR-BERT",
            "qata_resnet0524_cxr_qata_resnet50_cxr_native_keep_decoder_seed42",
            "text TGFS",
        ),
        (
            "ResNet50 + simple text",
            "qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42",
            "text TGFS",
        ),
        (
            "Scratch + simple text",
            "qata_paper0516_qata_simple_native_learned_both_seed42",
            "text TGFS",
        ),
    ]
    rows = []
    for label, run, group in specs:
        metric = load_metric(run)
        rows.append(
            {
                "label": label,
                "group": group,
                "run": run,
                "dice": metric["dice"],
                "iou": metric["iou"],
                "best_epoch": metric["epoch"],
                "threshold": metric["threshold"],
            }
        )

    labels = [r["label"] for r in rows]
    dice = [percent(r["dice"]) for r in rows]
    iou = [percent(r["iou"]) for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    colors = ["#b8c6a1", "#9dbb8e", "#d9a441", "#cf7a43", "#9e4f32"]
    bars = ax.bar(x, dice, width=0.68, color=colors, edgecolor="#262626", linewidth=0.7)
    ax.plot(x, iou, color="#1d3557", marker="o", linewidth=2.0, label="IoU")
    ax.set_title("QaTa-COV19: visual-only baselines vs. text-guided TGFS")
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(64, 86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    annotate_bars(ax, bars, dice)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.01,
        -0.28,
        "Bars: per-image Dice. Line: per-image IoU. Checkpoints selected by validation Dice.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    savefig(fig, "fig_qata_main_comparison")
    return rows


def plot_prompt_semantics() -> list[dict[str, object]]:
    panels = [
        (
            "Scratch simple, both fusion",
            [
                ("Native", "qata_diag0516_qata_simple_native_keep_both_seed42"),
                ("Empty", "qata_paper0516_qata_simple_empty_keep_both_seed42"),
                ("Shuffle", "qata_paper0516_qata_simple_shuffle_keep_both_seed42"),
                ("Generic", "qata_diag0516_qata_simple_generic_keep_both_seed42"),
            ],
        ),
        (
            "ResNet50 simple, decoder",
            [
                ("Native", "qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42"),
                ("Empty", "qata_resnet0523_fp32fix_qata_resnet50_simple_empty_keep_decoder_seed42"),
                ("Shuffle", "qata_resnet0523_qata_resnet50_simple_shuffle_keep_decoder_seed42"),
                ("Generic", "qata_resnet0523_qata_resnet50_simple_generic_keep_decoder_seed42"),
            ],
        ),
        (
            "Scratch CXR-BERT, both fusion",
            [
                ("Native", "screening0506_qata_cxr_frozen_keep_both_seed42"),
                ("Empty", "qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42"),
                ("Shuffle", "qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42"),
            ],
        ),
    ]

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.8), sharey=True)
    color_map = {
        "Native": "#3f7f5f",
        "Empty": "#c9b58a",
        "Shuffle": "#cc6f5a",
        "Generic": "#b58abf",
    }
    for ax, (title, items) in zip(axes, panels):
        labels, values = [], []
        native_value = None
        for label, run in items:
            metric = load_metric(run)
            value = percent(metric["dice"])
            if label == "Native":
                native_value = value
            labels.append(label)
            values.append(value)
            rows.append(
                {
                    "family": title,
                    "prompt": label.lower(),
                    "run": run,
                    "dice": metric["dice"],
                    "iou": metric["iou"],
                    "delta_dice_vs_native": None,
                }
            )
        bars = ax.bar(
            np.arange(len(labels)),
            values,
            color=[color_map[l] for l in labels],
            edgecolor="#222222",
            linewidth=0.6,
        )
        annotate_bars(ax, bars, values, dy=0.25)
        if native_value is not None:
            ax.axhline(native_value, color="#243b53", linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(73, 83.5)
    axes[0].set_ylabel("Per-image Dice (%)")
    fig.suptitle("Prompt semantics sanity check on QaTa-COV19", y=1.03, fontsize=13)
    fig.text(
        0.5,
        -0.04,
        "Native prompts consistently outperform empty, shuffled, or generic prompts.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    savefig(fig, "fig_qata_prompt_semantics")

    native_by_family = {}
    for row in rows:
        if row["prompt"] == "native":
            native_by_family[row["family"]] = row["dice"]
    for row in rows:
        native = native_by_family.get(row["family"])
        if native is not None and row["dice"] is not None:
            row["delta_dice_vs_native"] = float(row["dice"]) - float(native)
    return rows


def plot_frequency_ablation() -> list[dict[str, object]]:
    prior_panels = [
        (
            "Scratch simple, both fusion",
            [
                ("Keep HH", "qata_diag0516_qata_simple_native_keep_both_seed42"),
                ("Zero HH", "qata_paper0516_qata_simple_native_zero_both_seed42"),
                ("Learn HH", "qata_paper0516_qata_simple_native_learned_both_seed42"),
            ],
        ),
        (
            "ResNet50 simple, decoder",
            [
                ("Keep HH", "qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42"),
                ("Zero HH", "qata_resnet0523_qata_resnet50_simple_native_zero_decoder_seed42"),
                ("Learn HH", "qata_resnet0523_qata_resnet50_simple_native_learned_decoder_seed42"),
            ],
        ),
    ]
    drop_items = [
        ("None", "qata_resnet0523_fp32fix_qata_resnet50_simple_native_keep_decoder_seed42"),
        ("Drop LL", "qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_ll_decoder_seed42"),
        ("Drop LH", "qata_resnet0523_qata_resnet50_simple_native_drop_lh_decoder_seed42"),
        ("Drop HL", "qata_resnet0523_qata_resnet50_simple_native_drop_hl_decoder_seed42"),
        ("Drop HH", "qata_resnet0523_fp32fix_qata_resnet50_simple_native_drop_hh_decoder_seed42"),
    ]

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=False)
    prior_colors = ["#93a98c", "#d6a756", "#b95f40"]
    for ax, (title, items) in zip(axes[:2], prior_panels):
        labels, values = [], []
        for label, run in items:
            metric = load_metric(run)
            labels.append(label)
            values.append(percent(metric["dice"]))
            rows.append(
                {
                    "panel": title,
                    "variant": label,
                    "run": run,
                    "dice": metric["dice"],
                    "iou": metric["iou"],
                }
            )
        bars = ax.bar(np.arange(len(labels)), values, color=prior_colors, edgecolor="#222", linewidth=0.6)
        annotate_bars(ax, bars, values, dy=0.22)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=17, ha="right")
        ax.set_ylabel("Dice (%)")
        ax.set_ylim(80.5, 83.2)

    labels, values = [], []
    for label, run in drop_items:
        metric = load_metric(run)
        labels.append(label)
        values.append(percent(metric["dice"]))
        rows.append(
            {
                "panel": "ResNet50 drop one sub-band",
                "variant": label,
                "run": run,
                "dice": metric["dice"],
                "iou": metric["iou"],
            }
        )
    colors = ["#708d81", "#bd5d47", "#d9a441", "#5b8e9d", "#7c6a9e"]
    bars = axes[2].bar(np.arange(len(labels)), values, color=colors, edgecolor="#222", linewidth=0.6)
    annotate_bars(axes[2], bars, values, dy=0.22)
    axes[2].set_title("Drop one wavelet sub-band")
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels(labels, rotation=17, ha="right")
    axes[2].set_ylabel("Dice (%)")
    axes[2].set_ylim(80.0, 82.4)

    fig.suptitle("Frequency-prior and sub-band ablations on QaTa-COV19", y=1.04, fontsize=13)
    fig.text(
        0.5,
        -0.04,
        "Drop-band rows are useful mechanistic diagnostics; LL/HH FP32 reruns are most protocol-stable.",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    savefig(fig, "fig_qata_frequency_ablation")
    return rows


def plot_dual_metric_gap() -> list[dict[str, object]]:
    path = ROOT / "qata_dual_metrics_resnet_cxr_vs_best_ablation.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    labels = []
    values = []
    for row in rows:
        if "resnet" in row["run"] and "cxr" in row["run"]:
            labels.append("ResNet50+CXR\nper-image")
            values.append(100.0 * row["per_image_dice"])
            labels.append("ResNet50+CXR\nglobal")
            values.append(100.0 * row["global_dice"])
        else:
            labels.append("Best ablation\nper-image")
            values.append(100.0 * row["per_image_dice"])
            labels.append("Best ablation\nglobal")
            values.append(100.0 * row["global_dice"])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = ["#d6a756", "#9e6f2f", "#5b8e9d", "#2f6272"]
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="#222", linewidth=0.6)
    annotate_bars(ax, bars, values, dy=0.22)
    ax.set_title("Per-image Dice vs. global Dice on QaTa-COV19")
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(78, 92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(
        0.01,
        -0.23,
        "Global Dice pools pixels across the test set and is consistently higher here; paper protocol uses per-image mean.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    savefig(fig, "fig_qata_per_image_vs_global")
    return rows


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    generated = {}

    main_rows = plot_main_qata_comparison()
    prompt_rows = plot_prompt_semantics()
    freq_rows = plot_frequency_ablation()
    dual_rows = plot_dual_metric_gap()

    write_rows(OUT / "qata_main_comparison.csv", main_rows)
    write_rows(OUT / "qata_prompt_semantics.csv", prompt_rows)
    write_rows(OUT / "qata_frequency_ablation.csv", freq_rows)

    generated["figures"] = [
        "fig_qata_main_comparison.png",
        "fig_qata_prompt_semantics.png",
        "fig_qata_frequency_ablation.png",
        "fig_qata_per_image_vs_global.png",
    ]
    generated["tables"] = [
        "qata_main_comparison.csv",
        "qata_prompt_semantics.csv",
        "qata_frequency_ablation.csv",
    ]
    generated["dual_metric_source_rows"] = len(dual_rows)
    (OUT / "manifest.json").write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
