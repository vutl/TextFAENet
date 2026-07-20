from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDCLIP_ROOT = ROOT.parent / "MedCLIP-SAMv2"

# Reuse the checkpoint-compatible MedCLIP saliency wrapper and the common
# pixel-count evaluator used by the existing QaTa runner.
from run_medclipsamv2_qata_zeroshot import (  # noqa: E402
    evaluate_and_export,
    run_cmd,
    run_patched_saliency,
    run_sam_for_missing,
)


def read_prompt_rows(
    csv_path: Path,
    image_column: str,
    text_column: str | None,
) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if image_column not in reader.fieldnames:
            raise ValueError(
                f"Missing image column {image_column!r} in {csv_path}; "
                f"available={reader.fieldnames}"
            )
        resolved_text_column = text_column
        if resolved_text_column is None:
            resolved_text_column = next(
                (name for name in ("Description", "text", "prompt") if name in reader.fieldnames),
                None,
            )
        if resolved_text_column is None or resolved_text_column not in reader.fieldnames:
            raise ValueError(
                f"Cannot resolve text column in {csv_path}; available={reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            image_name = Path(str(row.get(image_column, "")).strip()).name
            prompt = str(row.get(resolved_text_column, "")).strip()
            if image_name and prompt:
                rows.append(
                    {
                        "image_name": image_name,
                        "mask_name": image_name,
                        "description": prompt,
                    }
                )
    if not rows:
        raise RuntimeError(f"No valid prompt rows found in {csv_path}")
    return rows


def count_matching_files(directory: Path, names: set[str]) -> int:
    return sum(1 for name in names if (directory / name).is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        "Run MedCLIP-SAMv2 zero-shot inference on a prompted folder dataset."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prompt-csv", type=Path, required=True)
    parser.add_argument("--image-subdir", default="test_images")
    parser.add_argument("--mask-subdir", default="test_masks")
    parser.add_argument("--image-column", default="Image")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--medclip-root", type=Path, default=DEFAULT_MEDCLIP_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-contours", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths/CSV mapping and write provenance files without inference.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    medclip_root = args.medclip_root.resolve()
    prompt_csv = args.prompt_csv.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    image_dir = dataset_root / args.image_subdir
    gt_dir = dataset_root / args.mask_subdir
    for path in (image_dir, gt_dir, prompt_csv, medclip_root):
        if not path.exists():
            raise FileNotFoundError(path)

    selected = read_prompt_rows(prompt_csv, args.image_column, args.text_column)
    selected = [
        row
        for row in selected
        if (image_dir / row["image_name"]).is_file()
        and (gt_dir / row["mask_name"]).is_file()
    ]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if not selected:
        raise RuntimeError("No prompt rows match both an image and a GT mask.")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_images = work_dir / "images"
    saliency_dir = work_dir / "saliency_maps"
    coarse_dir = work_dir / "coarse_masks"
    sam_dir = work_dir / "sam_masks_raw"
    for directory in (work_images, saliency_dir, coarse_dir, sam_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prompt_map: dict[str, str] = {}
    for row in selected:
        src = image_dir / row["image_name"]
        dst = work_images / row["image_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
        prompt_map[row["image_name"]] = row["description"]

    prompt_json = out_dir / "prompts_used.json"
    manifest_json = out_dir / "selected_cases.json"
    protocol_json = out_dir / "protocol.json"
    prompt_json.write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_json.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    protocol_json.write_text(
        json.dumps(
            {
                "model": "MedCLIP-SAMv2 zero-shot",
                "dataset_name": args.dataset_name,
                "dataset_root": str(dataset_root),
                "prompt_csv": str(prompt_csv),
                "prompt_format": "our structured prompt",
                "num_selected": len(selected),
                "num_contours": args.num_contours,
                "device": args.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.validate_only:
        print(
            json.dumps(
                {
                    "dataset_name": args.dataset_name,
                    "num_selected": len(selected),
                    "image_dir": str(image_dir),
                    "gt_dir": str(gt_dir),
                    "prompt_csv": str(prompt_csv),
                    "status": "validated",
                },
                indent=2,
            ),
            flush=True,
        )
        return

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

    names = set(prompt_map)
    if not args.skip_inference:
        run_patched_saliency(
            medclip_root, work_images, prompt_json, saliency_dir, args.device
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "medclipsamv2_kmeans_postprocess.py"),
                "--output-path",
                str(coarse_dir),
                "--sal-path",
                str(saliency_dir),
                "--num-contours",
                str(args.num_contours),
            ],
            cwd=ROOT,
            env=env,
        )
        run_sam_for_missing(
            medclip_root=medclip_root,
            image_dir=work_images,
            coarse_dir=coarse_dir,
            sam_dir=sam_dir,
            image_names=[row["image_name"] for row in selected],
            device=args.device,
            multicontour=args.num_contours > 1,
            env=env,
        )

    available = count_matching_files(sam_dir, names)
    if available != len(selected):
        raise RuntimeError(
            f"Incomplete SAM output: found {available}/{len(selected)} selected masks in {sam_dir}"
        )

    summary = evaluate_and_export(
        selected=selected,
        sam_dir=sam_dir,
        gt_dir=gt_dir,
        pred_mask_dir=out_dir / "pred_masks",
        output_csv=out_dir / "test_per_image_metrics.csv",
        output_json=out_dir / "summary.json",
    )
    summary["dataset_name"] = args.dataset_name
    summary["prompt_format"] = "our structured prompt"
    summary["prompt_csv"] = str(prompt_csv)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# MedCLIP-SAMv2 structured-prompt evaluation: {args.dataset_name}",
                "",
                f"- Samples: {summary['num_images']}",
                f"- Per-image Dice: {summary['per_image_dice']:.6f}",
                f"- Per-image IoU: {summary['per_image_iou']:.6f}",
                f"- Global Dice: {summary['global_dice']:.6f}",
                f"- Global IoU: {summary['global_iou']:.6f}",
                f"- Prompt CSV: `{prompt_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
