from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if (ROOT / ".hf_cache").exists():
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))

from src.models import FAENet, LFAENetTGFS, LFAENetTGFSv2


PALETTE = {
    "text": (28, 32, 38),
    "muted": (91, 101, 116),
    "bg": (255, 255, 255),
    "panel": (248, 250, 252),
    "green": (37, 146, 113),
    "blue": (58, 111, 183),
    "red": (224, 92, 76),
    "orange": (220, 145, 52),
    "purple": (126, 96, 157),
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[Path] = []
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
        ]
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(36, True)
FONT_SUBTITLE = font(20)
FONT_LABEL = font(18, True)
FONT_SMALL = font(15)
FONT_TINY = font(13)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def stable_token_id(token: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    value = 0
    for idx, ch in enumerate(token):
        value = (value + (idx + 1) * ord(ch)) % (vocab_size - 1)
    return value + 1


def simple_tokenize(texts: list[str], max_length: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
    attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
    for row_idx, text in enumerate(texts):
        tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
        ids = [stable_token_id(tok, vocab_size) for tok in tokens[:max_length]]
        if ids:
            input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, : len(ids)] = 1
    return input_ids, attention_mask


def build_model_from_config(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_type = cfg.get("model_type", "lfaenet_tgfs_v2")
    text_encoder_type = "biomedvlp-cxr-bert" if cfg.get("use_cxr_bert", True) else "simple"
    if model_type == "faenet":
        model = FAENet(in_channels=1, num_classes=1)
    elif model_type == "lfaenet_tgfs":
        model = LFAENetTGFS(
            in_channels=1,
            num_classes=1,
            text_dim=int(cfg.get("text_dim", 256)),
            vocab_size=int(cfg.get("vocab_size", 30522)),
            text_encoder_type=text_encoder_type,
            text_backbone_path=cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
            freeze_text_backbone=bool(cfg.get("freeze_text_backbone", True)),
        )
    elif model_type == "lfaenet_tgfs_v2":
        model = LFAENetTGFSv2(
            in_channels=1,
            num_classes=1,
            text_dim=int(cfg.get("text_dim", 256)),
            vocab_size=int(cfg.get("vocab_size", 30522)),
            text_encoder_type=text_encoder_type,
            text_backbone_path=cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
            freeze_text_backbone=bool(cfg.get("freeze_text_backbone", True)),
            drop_hh_in_decoder=bool(cfg.get("drop_hh_in_decoder", True)),
            hh_drop_mode=cfg.get("hh_drop_mode"),
            low_level_hf_scale=float(cfg.get("low_level_hf_scale", 0.6)),
            spatial_sharpen_power=float(cfg.get("spatial_sharpen_power", 2.0)),
            use_deep_supervision=bool(cfg.get("use_deep_supervision", False)),
            fusion_mode=cfg.get("fusion_mode", "decoder"),
            unfreeze_last_n=int(cfg.get("unfreeze_last_n", 0) or 0),
            lora_r=int(cfg.get("lora_r", 0) or 0),
            freeze_freq_gate=bool(cfg.get("freeze_freq_gate", False)),
            disable_spatial_mask=bool(cfg.get("disable_spatial_mask", False)),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model.to(device)


def build_tokenizer(cfg: dict[str, Any]):
    if cfg.get("model_type") == "faenet":
        return None
    if not cfg.get("use_cxr_bert", True):
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.get("cxr_bert_dir", "BiomedVLP-CXR-BERT-specialized"),
        trust_remote_code=True,
        local_files_only=True,
    )


def load_run(run_dir: Path, device: torch.device):
    cfg = read_json(run_dir / "config.json")
    if not isinstance(cfg, dict):
        raise FileNotFoundError(f"Missing or invalid config.json: {run_dir}")
    model = build_model_from_config(cfg, device)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best.pt: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    tokenizer = build_tokenizer(cfg)
    final = read_json(run_dir / "final_test.json") or {}
    threshold = float(final.get("best_threshold", ckpt.get("best_threshold", 0.5)))
    return model, tokenizer, cfg, threshold


def encode_texts(texts: list[str], tokenizer, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = int(cfg.get("max_text_len", 64))
    if tokenizer is not None:
        toks = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return toks["input_ids"], toks["attention_mask"]
    return simple_tokenize(texts, max_len, int(cfg.get("vocab_size", 30522)))


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    tokenizer,
    cfg: dict[str, Any],
    image: torch.Tensor,
    text: str,
    device: torch.device,
    threshold: float = 0.5,
    capture_debug: bool = False,
) -> dict[str, Any]:
    image_b = image.unsqueeze(0).to(device)
    if capture_debug and hasattr(model, "set_debug_capture"):
        model.set_debug_capture(True)
    if cfg.get("model_type") == "faenet":
        logits = model(image_b)
    else:
        ids, mask = encode_texts([text], tokenizer, cfg)
        logits = model(
            image_b,
            token_ids=ids.to(device),
            attention_mask=mask.to(device),
        )
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > threshold).astype(np.float32)
    debug = model.get_debug_outputs() if capture_debug and hasattr(model, "get_debug_outputs") else {}
    if capture_debug and hasattr(model, "set_debug_capture"):
        model.set_debug_capture(False)
    return {"prob": prob, "pred": pred, "debug": debug}


def sample_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - float(arr.min())
    den = float(arr.max() - arr.min())
    if den < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return (arr / den * 255.0).clip(0, 255).astype(np.uint8)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    g = (np.clip(image.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int] = PALETTE["green"]) -> np.ndarray:
    out = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def overlay_mask(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = PALETTE["red"],
    alpha: float = 0.45,
) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    canvas = np.ones((*gt.shape, 3), dtype=np.uint8) * 255
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    canvas[tp] = np.array([178, 224, 190], dtype=np.uint8)
    canvas[fp] = np.array([70, 130, 220], dtype=np.uint8)
    canvas[fn] = np.array([226, 68, 65], dtype=np.uint8)
    return canvas


def heatmap_red(mask: np.ndarray, base: np.ndarray | None = None, alpha: float = 0.48) -> np.ndarray:
    heat = to_uint8(mask).astype(np.float32) / 255.0
    if base is None:
        bg = np.ones((*mask.shape, 3), dtype=np.float32) * 255.0
    else:
        bg = base.astype(np.float32)
    red = np.zeros_like(bg)
    red[..., 0] = 255.0
    red[..., 1] = 40.0
    red[..., 2] = 20.0
    out = (1.0 - alpha * heat[..., None]) * bg + (alpha * heat[..., None]) * red
    return out.clip(0, 255).astype(np.uint8)


def resize_rgb(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB").resize((size, size), Image.BILINEAR)


def draw_tile(
    out: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: np.ndarray,
    xy: tuple[int, int],
    title: str,
    subtitle: str = "",
    tile: int = 176,
) -> None:
    x, y = xy
    draw.rounded_rectangle((x - 5, y - 34, x + tile + 5, y + tile + 30), radius=12, fill=(248, 250, 252), outline=(224, 228, 235), width=1)
    draw.text((x, y - 30), title, fill=PALETTE["text"], font=FONT_SMALL)
    if subtitle:
        draw.text((x, y - 13), subtitle, fill=PALETTE["muted"], font=FONT_TINY)
    out.paste(resize_rgb(img, tile), (x, y))


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width_chars: int,
    fill: tuple[int, int, int] = PALETTE["text"],
) -> None:
    import textwrap

    draw.multiline_text(xy, textwrap.fill(text, width=width_chars), fill=fill, font=FONT_TINY, spacing=4)


def spatial_mask_from_debug(debug: dict[str, Any], image_shape: tuple[int, int], stage: str = "dec1") -> np.ndarray | None:
    stage_debug = debug.get(stage)
    if not stage_debug or "spatial_mask" not in stage_debug:
        return None
    spatial = stage_debug["spatial_mask"][0:1]
    spatial = F.interpolate(spatial, size=image_shape, mode="bilinear", align_corners=False)
    return spatial.squeeze().detach().cpu().numpy()
