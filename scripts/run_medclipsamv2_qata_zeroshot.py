from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QATA_ROOT = ROOT.parent / "FMISeg" / "data" / "QaTa-COV19-v2"
DEFAULT_MEDCLIP_ROOT = ROOT.parent / "MedCLIP-SAMv2"


def read_qata_prompts(qata_root: Path, split: str) -> dict[str, dict[str, str]]:
    csv_path = qata_root / "prompt" / f"{split}.csv"
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mask_name = row["Image"].strip()
            if not mask_name.startswith("mask_"):
                continue
            image_name = mask_name[len("mask_") :]
            rows[mask_name] = {
                "mask_name": mask_name,
                "image_name": image_name,
                "description": row["Description"].strip(),
            }
    return rows


def load_case_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = []
    for item in data:
        name = item.get("mask_name") or item.get("Image") or item.get("image")
        if name:
            names.append(Path(str(name)).name)
    return names


def read_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)


def evaluate_and_export(
    selected: list[dict[str, str]],
    sam_dir: Path,
    gt_dir: Path,
    pred_mask_dir: Path,
    output_csv: Path,
    output_json: Path,
) -> dict[str, float | int | str]:
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_inter = total_union = total_pred = total_target = 0.0
    eps = 1e-6
    for rec in selected:
        image_name = rec["image_name"]
        mask_name = rec["mask_name"]
        pred_path = sam_dir / image_name
        if not pred_path.exists():
            print(f"Missing SAM output for {image_name}", flush=True)
            continue
        gt_path = gt_dir / mask_name
        gt_img = Image.open(gt_path).convert("L")
        gt = (np.asarray(gt_img, dtype=np.float32) > 127).astype(np.float32)
        pred = read_mask(pred_path, size=gt_img.size)
        pred_out_path = pred_mask_dir / mask_name
        Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(pred_out_path)

        inter = float((pred * gt).sum())
        pred_sum = float(pred.sum())
        target_sum = float(gt.sum())
        union = float(((pred + gt) > 0).sum())
        dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
        iou = (inter + eps) / (union + eps)
        rows.append(
            {
                "mask_name": mask_name,
                "dice": dice,
                "iou": iou,
                "intersection": inter,
                "union": union,
                "pred_pixels": pred_sum,
                "target_pixels": target_sum,
            }
        )
        total_inter += inter
        total_union += union
        total_pred += pred_sum
        total_target += target_sum

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mask_name", "dice", "iou", "intersection", "union", "pred_pixels", "target_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": len(rows),
        "per_image_dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "per_image_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
        "global_dice": float((2.0 * total_inter + eps) / (total_pred + total_target + eps)),
        "global_iou": float((total_inter + eps) / (total_union + eps)),
        "pred_dir": str(pred_mask_dir),
        "raw_sam_dir": str(sam_dir),
        "per_image_csv": str(output_csv),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_patched_saliency(
    medclip_root: Path,
    image_dir: Path,
    prompt_json: Path,
    saliency_dir: Path,
    device: str,
) -> None:
    import cv2
    import importlib.util
    import types
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    # MedCLIP-SAMv2 saliency code imports a top-level `scripts` package.
    # Put its saliency package before Text-FAENet/scripts to avoid collisions.
    medclip_root = medclip_root.resolve()
    saliency_pkg = medclip_root / "saliency_maps"
    for path in [str(saliency_pkg), str(medclip_root)]:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            module_file = getattr(sys.modules[key], "__file__", "") or ""
            if "Text-FAENet" in module_file:
                del sys.modules[key]

    def load_module(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {module_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = [str(saliency_pkg / "scripts")]  # type: ignore[attr-defined]
    sys.modules["scripts"] = scripts_pkg
    load_module("scripts.iba", saliency_pkg / "scripts" / "iba.py")
    freq_mod = load_module("scripts.freq_components", saliency_pkg / "scripts" / "freq_components.py")
    methods_mod = load_module("scripts.methods", saliency_pkg / "scripts" / "methods.py")
    vision_heatmap_freq_aware = methods_mod.vision_heatmap_freq_aware
    DWTForward = freq_mod.DWTForward
    SmartFusionBlock = freq_mod.SmartFusionBlock

    local_model = medclip_root / "saliency_maps" / "model"
    print("Loading MedCLIP-SAMv2 saliency model ...", flush=True)
    model = AutoModel.from_pretrained(str(local_model), trust_remote_code=True).to(device)
    try:
        processor = AutoProcessor.from_pretrained(str(local_model), trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(local_model), trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    model.eval()

    class CrossAttnWrapper(nn.Module):
        def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, query, key, value, **kwargs):
            return self.mha(query, key, value)

    class ShallowFusionFallback(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, dwt_feats, early_feats):
            return self.proj(dwt_feats)

    dwt_module = DWTForward().to(device)
    with torch.no_grad():
        dwt_out = dwt_module(torch.zeros(1, 3, 32, 32, device=device))
    in_ch = int(dwt_out.shape[1])
    fusion_block = SmartFusionBlock(hf_channels=in_ch, lf_channels=1, out_channels=32).to(device)
    cross_attn = CrossAttnWrapper(embed_dim=768, num_heads=12).to(device)
    attn_proj = nn.Linear(768, 1).to(device)
    shallow_fusion = ShallowFusionFallback(in_ch=in_ch, out_ch=in_ch).to(device)
    fusion_block.eval()
    cross_attn.eval()
    attn_proj.eval()
    shallow_fusion.eval()

    prompts = json.loads(prompt_json.read_text(encoding="utf-8"))
    saliency_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
    print(f"Generating MedCLIP-SAMv2 saliency maps: {len(image_paths)} images", flush=True)
    for image_path in tqdm(image_paths):
        out_path = saliency_dir / image_path.name
        if out_path.exists():
            continue
        caption = prompts[image_path.name]
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        text_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)], device=device)
        with torch.no_grad():
            vmap = vision_heatmap_freq_aware(
                text_ids,
                pixel_values,
                model,
                7,
                0.1,
                1.0,
                fusion_block,
                cross_attn,
                attn_proj,
                shallow_fusion,
                dwt_module,
                ensemble=False,
                progbar=False,
            )
        img_np = np.asarray(image)
        vmap_np = cv2.resize(np.asarray(vmap), (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        vmap_np = (np.clip(vmap_np, 0.0, 1.0) * 255).astype("uint8")
        cv2.imwrite(str(out_path), vmap_np)


def main() -> None:
    parser = argparse.ArgumentParser("Run MedCLIP-SAMv2 zero-shot inference on QaTa-COV19.")
    parser.add_argument("--medclip-root", type=Path, default=DEFAULT_MEDCLIP_ROOT)
    parser.add_argument("--qata-root", type=Path, default=DEFAULT_QATA_ROOT)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--case-json", type=Path, default=None, help="Optional JSON with mask_name entries to run only selected cases.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "external_metrics" / "medclipsamv2_qata_our_prompt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-contours", type=int, default=2)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qata_root = args.qata_root.resolve()
    medclip_root = args.medclip_root.resolve()
    split_dir = qata_root / ("Test" if args.split == "test" else "Train")
    image_dir = split_dir / "Images"
    gt_dir = split_dir / "GTs"

    all_rows = read_qata_prompts(qata_root, args.split)
    wanted = load_case_names(args.case_json)
    if wanted is None:
        selected = list(all_rows.values())
    else:
        selected = [all_rows[name] for name in wanted if name in all_rows]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if not selected:
        raise RuntimeError("No QaTa samples selected.")

    work_dir = args.out_dir / "work"
    work_images = work_dir / "images"
    saliency_dir = work_dir / "saliency_maps"
    coarse_dir = work_dir / "coarse_masks"
    sam_dir = work_dir / "sam_masks_raw"
    for directory in [work_images, saliency_dir, coarse_dir, sam_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    prompt_map = {}
    selected_manifest = []
    for rec in selected:
        src = image_dir / rec["image_name"]
        dst = work_images / rec["image_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
        prompt_map[rec["image_name"]] = rec["description"]
        selected_manifest.append(rec)

    prompt_json = args.out_dir / "qata_test_image_prompts.json"
    manifest_json = args.out_dir / "selected_cases.json"
    prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_json.write_text(json.dumps(selected_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))

    python_cmd = [sys.executable]
    if not args.skip_inference:
        run_patched_saliency(medclip_root, work_images, prompt_json, saliency_dir, args.device)
        run_cmd(
            python_cmd
            + [
                "postprocessing/postprocess_saliency_maps.py",
                "--input-path",
                str(work_images),
                "--output-path",
                str(coarse_dir),
                "--sal-path",
                str(saliency_dir),
                "--postprocess",
                "kmeans",
                "--filter",
                "--num-contours",
                str(args.num_contours),
            ],
            cwd=medclip_root,
            env=env,
        )
        sam_ckpt = medclip_root / "segment-anything" / "sam_checkpoints" / "sam_vit_h_4b8939.pth"
        run_cmd(
            python_cmd
            + [
                "segment-anything/prompt_sam.py",
                "--input",
                str(work_images),
                "--mask-input",
                str(coarse_dir),
                "--output",
                str(sam_dir),
                "--model-type",
                "vit_h",
                "--checkpoint",
                str(sam_ckpt),
                "--prompts",
                "boxes",
                "--multicontour",
                "--device",
                args.device,
            ],
            cwd=medclip_root,
            env=env,
        )

    summary = evaluate_and_export(
        selected=selected,
        sam_dir=sam_dir,
        gt_dir=gt_dir,
        pred_mask_dir=args.out_dir / "pred_masks",
        output_csv=args.out_dir / "test_per_image_metrics.csv",
        output_json=args.out_dir / "summary.json",
    )
    summary_md = args.out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# MedCLIP-SAMv2 QaTa Zero-Shot",
                "",
                f"- Samples: {summary['num_images']}",
                f"- Per-image Dice: {summary['per_image_dice']:.6f}",
                f"- Per-image IoU: {summary['per_image_iou']:.6f}",
                f"- Global Dice: {summary['global_dice']:.6f}",
                f"- Global IoU: {summary['global_iou']:.6f}",
                f"- Prediction masks: `{summary['pred_dir']}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
