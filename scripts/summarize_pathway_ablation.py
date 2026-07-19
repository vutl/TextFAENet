from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# variant_name -> (freq_gate driven by text?, spatial mask driven by text?)
PATHWAY_STATE = {
    "C0_full_tgfs": ("text", "text"),
    "C1_freeze_freq_gate": ("const", "text"),
    "C2_no_spatial_mask": ("text", "off"),
    "C3_no_pathways": ("const", "off"),
}


def load_seed_results(run_prefix: str, dataset: str, variant: str, seeds: list[int]) -> list[dict]:
    results = []
    for seed in seeds:
        run_dir = ROOT / "runs" / f"{run_prefix}_{dataset}_{variant}_seed{seed}"
        final_path = run_dir / "final_test.json"
        if not final_path.exists():
            print(f"  [missing] {final_path}")
            continue
        stats = json.loads(final_path.read_text(encoding="utf-8"))
        stats["seed"] = seed
        stats["run_dir"] = str(run_dir)
        results.append(stats)
    return results


def summarize(run_prefix: str, dataset: str, variants: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for variant in variants:
        seed_results = load_seed_results(run_prefix, dataset, variant, seeds)
        row = {
            "variant": variant,
            "freq_gate": PATHWAY_STATE.get(variant, ("?", "?"))[0],
            "spatial_mask": PATHWAY_STATE.get(variant, ("?", "?"))[1],
            "n_seeds": len(seed_results),
            "seeds_found": [r["seed"] for r in seed_results],
        }
        if seed_results:
            dice = np.array([r["dice"] for r in seed_results], dtype=np.float64) * 100.0
            iou = np.array([r["iou"] for r in seed_results], dtype=np.float64) * 100.0
            row["dice_mean"] = float(dice.mean())
            row["dice_std"] = float(dice.std())
            row["iou_mean"] = float(iou.mean())
            row["iou_std"] = float(iou.std())
        rows.append(row)
    return rows


def print_table(rows: list[dict], n_expected_seeds: int) -> None:
    print(f"\n{'Variant':<24} {'freq_gate':<10} {'spatial_mask':<13} {'seeds':<7} {'Dice':<16} {'IoU':<16}")
    print("-" * 92)
    for row in rows:
        seeds_str = f"{row['n_seeds']}/{n_expected_seeds}"
        if row["n_seeds"] == 0:
            print(f"{row['variant']:<24} {row['freq_gate']:<10} {row['spatial_mask']:<13} {seeds_str:<7} {'--':<16} {'--':<16}")
            continue
        dice_str = f"{row['dice_mean']:.2f} +/- {row['dice_std']:.2f}"
        iou_str = f"{row['iou_mean']:.2f} +/- {row['iou_std']:.2f}"
        flag = "" if row["n_seeds"] == n_expected_seeds else "  (incomplete)"
        print(f"{row['variant']:<24} {row['freq_gate']:<10} {row['spatial_mask']:<13} {seeds_str:<7} {dice_str:<16} {iou_str:<16}{flag}")

    complete = [r for r in rows if r["n_seeds"] > 0]
    by_name = {r["variant"]: r for r in complete}
    if all(v in by_name for v in ("C0_full_tgfs", "C1_freeze_freq_gate", "C3_no_pathways")):
        c0 = by_name["C0_full_tgfs"]["dice_mean"]
        c1 = by_name["C1_freeze_freq_gate"]["dice_mean"]
        c3 = by_name["C3_no_pathways"]["dice_mean"]
        print(f"\nPattern check: C0-C3 = {c0 - c3:+.2f} Dice, C1-C3 = {c1 - c3:+.2f} Dice")
        if c0 > c1 > c3 and (c1 - c3) < (c0 - c3):
            print("-> Consistent with the expected pattern: most of the text benefit flows through the frequency-gating pathway.")
        else:
            print("-> Does NOT match the expected C0>C1>C3 with (C1-C3)<(C0-C3) pattern; re-check before writing the paper claim.")


def save_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pathway_ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = ["variant", "freq_gate", "spatial_mask", "n_seeds", "dice_mean", "dice_std", "iou_mean", "iou_std"]
    with (out_dir / "pathway_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate Dice/IoU across seeds for the QaTa TGFS pathway ablation (Group C).")
    parser.add_argument("--run-prefix", type=str, default="qata_pathway")
    parser.add_argument("--dataset", type=str, default="qata")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["C0_full_tgfs", "C1_freeze_freq_gate", "C2_no_spatial_mask", "C3_no_pathways"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2026])
    parser.add_argument("--out-dir", type=str, default="runs/qata_pathway_summary")
    args = parser.parse_args()

    rows = summarize(args.run_prefix, args.dataset, args.variants, args.seeds)
    print_table(rows, n_expected_seeds=len(args.seeds))
    save_outputs(rows, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
