from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEDCLIP_ROOT = ROOT.parent / "MedCLIP-SAMv2"


def read_prompt_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Image" not in reader.fieldnames:
            raise ValueError(f"Expected Image column in {path}")
        text_column = next(
            (name for name in ("Description", "text", "prompt") if name in reader.fieldnames),
            None,
        )
        if text_column is None:
            raise ValueError(f"No prompt column found in {path}")
        return [
            {
                "Image": Path(str(row["Image"])).name,
                "Description": str(row[text_column]).strip(),
            }
            for row in reader
            if str(row.get("Image", "")).strip() and str(row.get(text_column, "")).strip()
        ]


def prepare_dataset(dataset: str, output_root: Path) -> None:
    data_root = ROOT / "datasets" / dataset
    prompt_root = MEDCLIP_ROOT / "data" / dataset
    dataset_out = output_root / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        prompt_csv = prompt_root / f"FMISeg_{split}" / f"{split}.csv"
        image_dir = data_root / f"{split}_images"
        mask_dir = data_root / f"{split}_masks"
        rows = read_prompt_csv(prompt_csv)
        prepared: list[dict[str, str]] = []
        missing: list[str] = []
        for row in rows:
            name = row["Image"]
            image_path = image_dir / name
            mask_path = mask_dir / name
            if not image_path.is_file() or not mask_path.is_file():
                missing.append(name)
                continue
            prepared.append(
                {
                    "image_path": str(Path(f"{split}_images") / name),
                    "mask_path": str(Path(f"{split}_masks") / name),
                    "Description": row["Description"],
                    "Image": name,
                }
            )
        if missing or len(prepared) != len(rows):
            raise RuntimeError(
                f"{dataset}/{split}: matched {len(prepared)}/{len(rows)}; "
                f"missing examples={missing[:10]}"
            )
        output_csv = dataset_out / f"{split}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_path", "mask_path", "Description", "Image"],
            )
            writer.writeheader()
            writer.writerows(prepared)
        print(f"Wrote {output_csv}: {len(prepared)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        "Prepare path-explicit FMISeg CSVs using MedCLIP-SAMv2-style prompts."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["brain_tumors", "breast_tumors"],
        default=["brain_tumors", "breast_tumors"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "external_runs" / "fmiseg_prompt_csvs",
    )
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    for dataset in args.datasets:
        prepare_dataset(dataset, output_root)


if __name__ == "__main__":
    main()
