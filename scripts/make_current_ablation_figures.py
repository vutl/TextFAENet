from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


RUNS: list[dict[str, str]] = [
    {
        "key": "simple_native_keep_both",
        "label": "Simple native, HH keep, both",
        "short": "Simple native both",
        "path": "runs/qata_diag0516_qata_simple_native_keep_both_seed42",
        "family": "simple",
        "status": "complete",
    },
    {
        "key": "cxr_frozen_keep_both",
        "label": "CXR-BERT native, HH keep, both",
        "short": "CXR native both",
        "path": "runs/screening0506_qata_cxr_frozen_keep_both_seed42",
        "family": "cxr",
        "status": "complete",
    },
    {
        "key": "cxr_frozen_keep_decoder",
        "label": "CXR-BERT native, HH keep, decoder",
        "short": "CXR native decoder",
        "path": "runs/screening0506_qata_cxr_frozen_keep_decoder_seed42",
        "family": "cxr",
        "status": "complete",
    },
    {
        "key": "cxr_frozen_learned_both",
        "label": "CXR-BERT native, HH learned, both",
        "short": "CXR learned both",
        "path": "runs/qata_diag0516_qata_cxr_frozen_learned_both_seed42",
        "family": "cxr",
        "status": "complete",
    },
    {
        "key": "cxr_empty_keep_both",
        "label": "CXR-BERT empty prompt, HH keep, both",
        "short": "CXR empty both",
        "path": "runs/qata_diag0516_qata_cxr_frozen_keep_both_empty_seed42",
        "family": "perturb",
        "status": "complete",
    },
    {
        "key": "cxr_shuffle_keep_both",
        "label": "CXR-BERT shuffled prompt, HH keep, both",
        "short": "CXR shuffle both",
        "path": "runs/qata_diag0516_qata_cxr_frozen_keep_both_shuffle_seed42",
        "family": "perturb",
        "status": "complete",
    },
    {
        "key": "simple_generic_keep_both",
        "label": "Simple generic, HH keep, both",
        "short": "Simple generic both",
        "path": "runs/qata_diag0516_qata_simple_generic_keep_both_seed42",
        "family": "simple_bad",
        "status": "complete",
    },
    {
        "key": "legacy_cxr_zero_decoder",
        "label": "Legacy CXR-BERT hard HH zero/drop",
        "short": "Legacy CXR zero",
        "path": "runs/qata_b4_e50_cxrbert_frozen_v2",
        "family": "legacy",
        "status": "complete",
    },
    {
        "key": "legacy_cxr_zero_decoder_rerun",
        "label": "Legacy CXR-BERT hard HH zero/drop rerun",
        "short": "Legacy CXR rerun",
        "path": "runs/qata_b4_e50_cxrbert_frozen_v2_rerun",
        "family": "legacy",
        "status": "complete",
    },
    {
        "key": "faenet_visual_valclean",
        "label": "FAENet visual-only, older SGD/poly",
        "short": "FAENet visual",
        "path": "runs/qata_faenet_notext_valclean_e5",
        "family": "visual",
        "status": "complete",
    },
    {
        "key": "faenet_visual_adamw",
        "label": "FAENet visual-only, older AdamW/cosine",
        "short": "FAENet AdamW",
        "path": "runs/qata_faenet_notext_adamw_cosine_e30",
        "family": "visual",
        "status": "complete",
    },
    {
        "key": "cxr_lora8_keep_both",
        "label": "CXR-BERT LoRA-r8 native, HH keep, both",
        "short": "CXR LoRA-r8",
        "path": "runs/qata_diag0516_qata_cxr_lora8_keep_both_seed42",
        "family": "incomplete",
        "status": "incomplete",
    },
]


PALETTE = {
    "simple": (34, 142, 112),
    "simple_bad": (94, 170, 146),
    "cxr": (54, 105, 176),
    "perturb": (224, 111, 89),
    "legacy": (118, 91, 140),
    "visual": (135, 146, 166),
    "incomplete": (190, 190, 190),
    "grid": (224, 228, 235),
    "text": (28, 32, 38),
    "muted": (93, 102, 115),
    "bg": (255, 255, 255),
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ]
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(42, bold=True)
FONT_SUBTITLE = font(24)
FONT_LABEL = font(24)
FONT_LABEL_BOLD = font(24, bold=True)
FONT_SMALL = font(19)
FONT_SMALL_BOLD = font(19, bold=True)
FONT_TINY = font(15)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def best_history(history: Any) -> dict[str, Any] | None:
    if isinstance(history, dict):
        history = history.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    valid = [row for row in history if isinstance(row, dict) and row.get("val_dice") is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row.get("val_dice", float("-inf"))))


def last_history(history: Any) -> dict[str, Any] | None:
    if isinstance(history, dict):
        history = history.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    return history[-1]


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in RUNS:
        run_dir = ROOT / spec["path"]
        cfg = read_json(run_dir / "config.json") or {}
        final = read_json(run_dir / "final_test.json") or {}
        hist = read_json(run_dir / "history.json") or []
        best = best_history(hist) or {}
        last = last_history(hist) or {}
        records.append(
            {
                **spec,
                "run_dir": str(run_dir),
                "exists": run_dir.exists(),
                "final_exists": bool(final),
                "test_loss": final.get("loss"),
                "test_iou": final.get("iou"),
                "test_dice": final.get("dice"),
                "best_epoch": final.get("best_epoch", best.get("epoch")),
                "best_threshold": final.get("best_threshold", best.get("val_threshold")),
                "best_val_dice": best.get("val_dice"),
                "best_val_iou": best.get("val_iou"),
                "best_val_loss": best.get("val_loss"),
                "epochs_logged": len(hist) if isinstance(hist, list) else len(hist.get("history", [])) if isinstance(hist, dict) else 0,
                "last_epoch": last.get("epoch") if isinstance(last, dict) else None,
                "last_train_dice": last.get("train_dice") if isinstance(last, dict) else None,
                "last_val_dice": last.get("val_dice") if isinstance(last, dict) else None,
                "model_type": cfg.get("model_type"),
                "use_cxr_bert": cfg.get("use_cxr_bert"),
                "prompt_mode": cfg.get("prompt_mode"),
                "hh_drop_mode": cfg.get("hh_drop_mode"),
                "fusion_mode": cfg.get("fusion_mode"),
                "lora_r": cfg.get("lora_r"),
            }
        )
    return records


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def save_summary_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "key",
        "label",
        "path",
        "status",
        "epochs_logged",
        "best_epoch",
        "best_threshold",
        "best_val_dice",
        "test_loss",
        "test_iou",
        "test_dice",
        "model_type",
        "use_cxr_bert",
        "prompt_mode",
        "hh_drop_mode",
        "fusion_mode",
        "lora_r",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int],
    font_obj: ImageFont.ImageFont,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font_obj, spacing=4, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - w) // 2
    y = box[1] + (box[3] - box[1] - h) // 2
    draw.multiline_text((x, y), text, fill=fill, font=font_obj, spacing=4, align="center")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=(55, 65, 81), width: int = 4) -> None:
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) > abs(ey - sy):
        sign = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - 16 * sign, ey - 9), (ex - 16 * sign, ey + 9)]
    else:
        sign = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 9, ey - 16 * sign), (ex + 9, ey - 16 * sign)]
    draw.polygon(points, fill=fill)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = (70, 80, 95),
    text_fill: tuple[int, int, int] = PALETTE["text"],
    font_obj: ImageFont.ImageFont = FONT_LABEL_BOLD,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)
    draw_centered_text(draw, box, text, text_fill, font_obj)


def make_tgfs_schematic(out_path: Path) -> None:
    w, h = 1900, 1480
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((80, 45), "TGFSBlockV2: Text-Guided Frequency Selection", fill=PALETTE["text"], font=FONT_TITLE)
    draw.text(
        (80, 98),
        "Feature-level DWT with text-conditioned sub-band gates and token-grounded spatial masking.",
        fill=PALETTE["muted"],
        font=FONT_SUBTITLE,
    )

    main_x = 190
    box_w = 330
    box_h = 76
    y0 = 175
    gap = 34
    boxes = [
        ("Input feature X", (230, 240, 255)),
        ("Local conv\npre-refine", (229, 246, 238)),
        ("Haar DWT", (236, 242, 255)),
        ("LL / LH / HL / HH", (245, 239, 255)),
        ("Per-band ICCA", (234, 250, 241)),
        ("Text-guided gates\nalpha_LL / alpha_LH\nalpha_HL / alpha_HH", (255, 244, 230)),
        ("CCCA\ncross-band interaction", (235, 247, 255)),
        ("Token-grounded\nspatial mask M_s", (255, 239, 234)),
        ("Mixer + optional\nspatial self-attn", (244, 244, 244)),
        ("Inverse DWT", (236, 242, 255)),
        ("Refined output", (228, 247, 238)),
    ]

    centers: list[tuple[int, int]] = []
    for idx, (text, color) in enumerate(boxes):
        y = y0 + idx * (box_h + gap)
        box = (main_x, y, main_x + box_w, y + box_h)
        rounded_box(draw, box, text, fill=color, font_obj=FONT_SMALL_BOLD if idx in {5, 7} else FONT_LABEL_BOLD)
        centers.append((main_x + box_w // 2, y + box_h // 2))
        if idx > 0:
            prev_y = y0 + (idx - 1) * (box_h + gap) + box_h
            arrow(draw, (main_x + box_w // 2, prev_y + 4), (main_x + box_w // 2, y - 4))

    text_x = 760
    text_boxes = [
        ("Prompt tokens", (text_x, 265, text_x + 310, 345), (255, 249, 226)),
        ("Text encoder\nSimple GRU or CXR-BERT", (text_x, 390, text_x + 310, 486), (255, 249, 226)),
        ("Pooled text vector", (text_x, 545, text_x + 310, 625), (255, 244, 230)),
        ("Token embeddings", (text_x, 700, text_x + 310, 780), (255, 244, 230)),
    ]
    for i, (text, box, color) in enumerate(text_boxes):
        rounded_box(draw, box, text, fill=color, outline=(196, 139, 53), font_obj=FONT_SMALL_BOLD)
        if i > 0:
            prev = text_boxes[i - 1][1]
            arrow(draw, ((prev[0] + prev[2]) // 2, prev[3] + 6), ((box[0] + box[2]) // 2, box[1] - 6), fill=(150, 103, 37), width=4)

    # Text branch connections into the main TGFS path.
    gate_y = y0 + 5 * (box_h + gap) + box_h // 2
    mask_y = y0 + 7 * (box_h + gap) + box_h // 2
    arrow(draw, (text_x, 585), (main_x + box_w + 8, gate_y), fill=(150, 103, 37), width=4)
    arrow(draw, (text_x, 740), (main_x + box_w + 8, mask_y), fill=(150, 103, 37), width=4)
    draw.text((560, 585), "channel gates", fill=(125, 83, 25), font=FONT_SMALL_BOLD)
    draw.text((560, 790), "visual-query / text-key attention", fill=(125, 83, 25), font=FONT_SMALL_BOLD)

    # Detail callout.
    callout = (1190, 250, 1800, 910)
    draw.rounded_rectangle(callout, radius=24, fill=(249, 250, 252), outline=(197, 204, 216), width=3)
    draw.text((1230, 285), "What the block claims", fill=PALETTE["text"], font=FONT_LABEL_BOLD)
    bullets = [
        "1. Text does not just concatenate with a feature map.",
        "2. Pooled text gates LL/LH/HL/HH separately.",
        "3. Token embeddings form a spatial grounding mask.",
        "4. HH can be zero, kept, or learned-scaled.",
        "5. Decoder stages can expose gate and mask debug maps.",
    ]
    y = 340
    for item in bullets:
        draw.text((1230, y), item, fill=PALETTE["text"], font=FONT_SMALL)
        y += 58

    draw.text(
        (1230, 760),
        "Current implementation is feature-level DWT.\nIt is not yet the raw LF/HF dual-branch hybrid.",
        fill=(150, 58, 49),
        font=FONT_SMALL_BOLD,
        spacing=7,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def value_range(values: list[float], forced_min: float | None = None, forced_max: float | None = None) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if forced_min is not None:
        low = forced_min
    else:
        low = max(0.0, math.floor((low - 0.02) * 20) / 20)
    if forced_max is not None:
        high = forced_max
    else:
        high = min(1.0, math.ceil((high + 0.02) * 20) / 20)
    if high <= low:
        high = low + 0.1
    return low, high


def draw_hbar(
    records: list[dict[str, Any]],
    out_path: Path,
    title: str,
    subtitle: str,
    metric: str = "test_dice",
    forced_min: float | None = None,
    forced_max: float | None = None,
    sort: bool = False,
) -> None:
    rows = [r for r in records if finite_float(r.get(metric)) is not None]
    if sort:
        rows = sorted(rows, key=lambda r: float(r[metric]), reverse=True)
    values = [float(r[metric]) for r in rows]
    x_min, x_max = value_range(values, forced_min, forced_max)

    left_w = 560
    plot_w = 900
    right_w = 220
    top = 165
    row_h = 72
    bottom = 105
    w = left_w + plot_w + right_w + 110
    h = top + row_h * len(rows) + bottom
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((55, 38), title, fill=PALETTE["text"], font=FONT_TITLE)
    draw.text((55, 92), subtitle, fill=PALETTE["muted"], font=FONT_SUBTITLE)

    x0 = left_w
    x1 = left_w + plot_w
    y_axis_top = top - 18
    y_axis_bottom = top + row_h * len(rows) - 18

    ticks = [x_min + i * (x_max - x_min) / 5 for i in range(6)]
    for tick in ticks:
        x = x0 + int((tick - x_min) / (x_max - x_min) * plot_w)
        draw.line([(x, y_axis_top), (x, y_axis_bottom)], fill=PALETTE["grid"], width=2)
        draw.text((x - 22, y_axis_bottom + 18), f"{tick:.2f}", fill=PALETTE["muted"], font=FONT_TINY)
    draw.line([(x0, y_axis_bottom), (x1, y_axis_bottom)], fill=(170, 178, 190), width=2)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        value = float(row[metric])
        color = PALETTE.get(row.get("family", "cxr"), (70, 120, 180))
        label = str(row.get("label", row["key"]))
        draw.text((55, y + 8), label, fill=PALETTE["text"], font=FONT_LABEL_BOLD)
        detail = f"best epoch {row.get('best_epoch', 'n/a')}, val {finite_float(row.get('best_val_dice')) or 0:.3f}" if row.get("best_val_dice") is not None else "incomplete"
        draw.text((55, y + 39), detail, fill=PALETTE["muted"], font=FONT_TINY)
        bar_x0 = x0
        bar_x1 = x0 + int((value - x_min) / (x_max - x_min) * plot_w)
        draw.rounded_rectangle((bar_x0, y + 10, bar_x1, y + 48), radius=14, fill=color)
        draw.text((bar_x1 + 14, y + 15), f"{value:.4f}", fill=PALETTE["text"], font=FONT_SMALL_BOLD)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_panel_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    items: list[tuple[str, float, tuple[int, int, int]]],
    y_min: float,
    y_max: float,
    note: str = "",
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=22, fill=(249, 250, 252), outline=(220, 224, 232), width=2)
    draw.text((x0 + 26, y0 + 20), title, fill=PALETTE["text"], font=FONT_LABEL_BOLD)
    if note:
        draw.text((x0 + 26, y0 + 52), note, fill=PALETTE["muted"], font=FONT_TINY)

    chart_left = x0 + 70
    chart_right = x1 - 30
    chart_top = y0 + 92
    chart_bottom = y1 - 72
    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill=(170, 178, 190), width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(170, 178, 190), width=2)

    tick_fmt = "{:.3f}" if (y_max - y_min) < 0.05 else "{:.2f}"
    for frac in [0.0, 0.5, 1.0]:
        val = y_min + frac * (y_max - y_min)
        y = chart_bottom - int(frac * (chart_bottom - chart_top))
        draw.line([(chart_left, y), (chart_right, y)], fill=PALETTE["grid"], width=1)
        draw.text((x0 + 18, y - 9), tick_fmt.format(val), fill=PALETTE["muted"], font=FONT_TINY)

    n = len(items)
    slot = (chart_right - chart_left) / max(n, 1)
    bar_w = min(92, int(slot * 0.56))
    for idx, (label, value, color) in enumerate(items):
        cx = int(chart_left + slot * idx + slot / 2)
        y_val = chart_bottom - int((value - y_min) / (y_max - y_min) * (chart_bottom - chart_top))
        y_val = max(chart_top, min(chart_bottom, y_val))
        draw.rounded_rectangle((cx - bar_w // 2, y_val, cx + bar_w // 2, chart_bottom), radius=10, fill=color)
        draw.text((cx - 38, y_val - 27), f"{value:.3f}", fill=PALETTE["text"], font=FONT_TINY)
        draw.multiline_text((cx - 74, chart_bottom + 14), label, fill=PALETTE["text"], font=FONT_TINY, spacing=2, align="center")


def by_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["key"]): r for r in records}


def make_family_comparison(records: list[dict[str, Any]], out_path: Path) -> None:
    rec = by_key(records)
    w, h = 1750, 1220
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 45), "QaTa Diagnostic Comparisons", fill=PALETTE["text"], font=FONT_TITLE)
    draw.text(
        (70, 98),
        "Small panels isolate text semantics, fusion locus, text encoder choice, and HH prior.",
        fill=PALETTE["muted"],
        font=FONT_SUBTITLE,
    )

    def dice(key: str) -> float:
        value = finite_float(rec[key].get("test_dice"))
        if value is None:
            raise ValueError(f"Missing test_dice for {key}")
        return value

    panel_w = 780
    panel_h = 475
    boxes = [
        (70, 175, 70 + panel_w, 175 + panel_h),
        (900, 175, 900 + panel_w, 175 + panel_h),
        (70, 700, 70 + panel_w, 700 + panel_h),
        (900, 700, 900 + panel_w, 700 + panel_h),
    ]

    draw_panel_bars(
        draw,
        boxes[0],
        "Text semantics: same CXR family",
        [
            ("native", dice("cxr_frozen_keep_both"), PALETTE["cxr"]),
            ("empty", dice("cxr_empty_keep_both"), PALETTE["perturb"]),
            ("shuffle", dice("cxr_shuffle_keep_both"), PALETTE["perturb"]),
        ],
        y_min=0.74,
        y_max=0.83,
        note="Correct prompt is the only changed semantic signal.",
    )
    draw_panel_bars(
        draw,
        boxes[1],
        "Fusion locus: CXR frozen",
        [
            ("decoder", dice("cxr_frozen_keep_decoder"), PALETTE["cxr"]),
            ("both", dice("cxr_frozen_keep_both"), PALETTE["cxr"]),
        ],
        y_min=0.805,
        y_max=0.820,
        note="Encoder+decoder text injection gives a small gain.",
    )
    draw_panel_bars(
        draw,
        boxes[2],
        "Text encoder: native + keep + both",
        [
            ("CXR-BERT\nfrozen", dice("cxr_frozen_keep_both"), PALETTE["cxr"]),
            ("simple\nGRU", dice("simple_native_keep_both"), PALETTE["simple"]),
        ],
        y_min=0.805,
        y_max=0.825,
        note="Simple encoder is currently the best completed QaTa run.",
    )
    draw_panel_bars(
        draw,
        boxes[3],
        "HH prior: CXR family",
        [
            ("legacy\nzero", dice("legacy_cxr_zero_decoder"), PALETTE["legacy"]),
            ("keep", dice("cxr_frozen_keep_both"), PALETTE["cxr"]),
            ("learned", dice("cxr_frozen_learned_both"), PALETTE["cxr"]),
        ],
        y_min=0.805,
        y_max=0.820,
        note="Legacy zero is not perfectly protocol-matched.",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser("Create current Text-FAENet architecture and ablation figures.")
    parser.add_argument("--out-dir", type=str, default="paper_figures/current_ablation")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    save_summary_csv(records, out_dir / "qata_ablation_summary.csv")

    summary_keys = [
        "simple_native_keep_both",
        "cxr_frozen_keep_both",
        "cxr_frozen_keep_decoder",
        "cxr_empty_keep_both",
        "cxr_shuffle_keep_both",
        "simple_generic_keep_both",
    ]
    rec_map = by_key(records)
    summary_records = [rec_map[k] for k in summary_keys]
    completed_records = [r for r in records if finite_float(r.get("test_dice")) is not None]

    make_tgfs_schematic(out_dir / "fig_tgfs_module_schematic.png")
    draw_hbar(
        summary_records,
        out_dir / "fig_qata_ablation_summary.png",
        title="QaTa Ablation Summary",
        subtitle="Final test Dice for prompt, encoder, and fusion variants already available.",
        forced_min=0.74,
        forced_max=0.83,
        sort=False,
    )
    draw_hbar(
        completed_records,
        out_dir / "fig_qata_full_ablation_ranking.png",
        title="QaTa Completed Run Ranking",
        subtitle="All completed runs from the current report; legacy and visual-only rows are included for context.",
        forced_min=0.50,
        forced_max=0.84,
        sort=True,
    )
    make_family_comparison(records, out_dir / "fig_qata_diagnostic_panels.png")

    manifest = {
        "out_dir": str(out_dir),
        "figures": [
            "fig_tgfs_module_schematic.png",
            "fig_qata_ablation_summary.png",
            "fig_qata_full_ablation_ranking.png",
            "fig_qata_diagnostic_panels.png",
            "qata_ablation_summary.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved current ablation figures to: {out_dir}")
    for name in manifest["figures"]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
