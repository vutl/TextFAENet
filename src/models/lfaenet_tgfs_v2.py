from __future__ import annotations

"""
LFAENet-TGFS v2
----------------
Patch focused on fixing the main design bug in the first TGFS version:

1) text was only used as a global pooled channel gate
2) the residual path let the network ignore text too easily

This version adds:
- token-level text features
- pooled text gating for LL/LH/HL/HH
- lightweight text-to-spatial grounding on the aggregated frequency feature
- text-conditioned scaling for BOTH identity and reconstructed branches

Expected use:
- replace your current lfaenet_tgfs.py with this file
- keep train_qata.py mostly unchanged, because it already passes token_ids + attention_mask
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class SpatialSelfAttention2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 8)
        self.q = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj(out)
        return x + self.gamma * out


def haar_dwt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    b, c, h, w = ll.shape
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    out = torch.zeros((b, c, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out


class ICCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = x.mean(dim=(2, 3))
        a = torch.sigmoid(self.mlp(z)).view(b, c, 1, 1)
        return x * a


class CCCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mix = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 4),
        )

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor):
        b, c, _, _ = ll.shape
        g = torch.cat([
            ll.mean(dim=(2, 3)),
            lh.mean(dim=(2, 3)),
            hl.mean(dim=(2, 3)),
            hh.mean(dim=(2, 3)),
        ], dim=1)
        w = torch.sigmoid(self.mix(g))
        w_ll, w_lh, w_hl, w_hh = torch.chunk(w, 4, dim=1)
        ll = ll * w_ll.view(b, c, 1, 1)
        lh = lh * w_lh.view(b, c, 1, 1)
        hl = hl * w_hl.view(b, c, 1, 1)
        hh = hh * w_hh.view(b, c, 1, 1)
        return ll, lh, hl, hh


class FreqA(nn.Module):
    def __init__(self, channels: int, use_attention: bool = False) -> None:
        super().__init__()
        self.icca_ll = ICCA(channels)
        self.icca_lh = ICCA(channels)
        self.icca_hl = ICCA(channels)
        self.icca_hh = ICCA(channels)
        self.ccca = CCCA(channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
        )
        self.attn = SpatialSelfAttention2D(channels * 4) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar_dwt2d(x)
        ll = self.icca_ll(ll)
        lh = self.icca_lh(lh)
        hl = self.icca_hl(hl)
        hh = self.icca_hh(hh)
        ll, lh, hl, hh = self.ccca(ll, lh, hl, hh)
        agg = torch.cat([ll, lh, hl, hh], dim=1)
        agg = self.mix(agg)
        agg = self.attn(agg)
        ll, lh, hl, hh = torch.chunk(agg, 4, dim=1)
        rec = haar_idwt2d(ll, lh, hl, hh)
        return x + rec


class EncoderStagePlain(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.freqa = FreqA(out_channels, use_attention=use_attention)

    def forward(
        self,
        x: torch.Tensor,
        text_pooled: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del text_pooled, text_tokens, text_mask
        return self.freqa(self.block(x))


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=pad_id)
        self.gru = nn.GRU(text_dim, text_dim, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(text_dim, text_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            mask = token_ids.ne(self.pad_id).float()
        else:
            mask = attention_mask.float()
        x = self.embedding(token_ids)
        x, _ = self.gru(x)
        x = self.proj(x)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return x, pooled


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = float(alpha if alpha is not None else rank) / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def _bert_encoder_layers(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "bert", model)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if isinstance(layers, (nn.ModuleList, list, tuple)):
        return list(layers)
    return []


def _inject_qv_lora(module: nn.Module, rank: int) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in {"query", "value"}:
            setattr(module, name, LoRALinear(child, rank=rank))
            replaced += 1
        else:
            replaced += _inject_qv_lora(child, rank=rank)
    return replaced


class CXRBertTextEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str,
        out_dim: int,
        freeze: bool = True,
        use_mean_pool: bool = True,
        unfreeze_last_n: int = 0,
        lora_r: int = 0,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Text backbone directory not found: {model_dir}")
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        hidden = int(getattr(cfg, "hidden_size", 768))
        self.token_proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim)
        self.use_mean_pool = use_mean_pool
        self.unfreeze_last_n = max(0, int(unfreeze_last_n))
        self.lora_r = max(0, int(lora_r))
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        layers = _bert_encoder_layers(self.model)
        if self.unfreeze_last_n > 0:
            if not layers:
                raise ValueError("--unfreeze-last-n was set, but encoder layers could not be found in CXR-BERT.")
            for layer in layers[-self.unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if self.lora_r > 0:
            target_layers = layers[-self.unfreeze_last_n:] if self.unfreeze_last_n > 0 else layers
            target_modules = target_layers if target_layers else [self.model]
            self.lora_replaced = sum(_inject_qv_lora(layer, rank=self.lora_r) for layer in target_modules)
            if self.lora_replaced == 0:
                raise ValueError("LoRA requested, but no attention query/value Linear layers were found.")
        else:
            self.lora_replaced = 0

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            attention_mask = token_ids.ne(0).long()
        if hasattr(self.model, "bert"):
            outputs = self.model.bert(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
        else:
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
        tokens = self.token_proj(outputs.last_hidden_state)
        if self.use_mean_pool:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = tokens[:, 0, :]
        return tokens, pooled


class TGFSBlockV2(nn.Module):
    def __init__(
        self,
        channels: int,
        text_dim: int,
        use_attention: bool = False,
        drop_hh: bool = False,
        hh_drop_mode: str | None = None,
        lh_hl_scale: float = 1.0,
        spatial_sharpen_power: float = 2.0,
        freeze_freq_gate: bool = False,
        disable_spatial_mask: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        if hh_drop_mode is None:
            hh_drop_mode = "zero" if drop_hh else "keep"
        if hh_drop_mode not in {"zero", "keep", "learned"}:
            raise ValueError(f"Unsupported hh_drop_mode: {hh_drop_mode}")
        self.hh_drop_mode = hh_drop_mode
        self.drop_hh = hh_drop_mode == "zero"
        self.lh_hl_scale = lh_hl_scale
        self.spatial_sharpen_power = spatial_sharpen_power
        self.freeze_freq_gate = freeze_freq_gate
        self.disable_spatial_mask = disable_spatial_mask
        if hh_drop_mode == "learned":
            self.hh_scale_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter("hh_scale_logit", None)
        self.capture_debug = False
        self.last_debug: Optional[dict[str, torch.Tensor]] = None
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.icca_ll = ICCA(channels)
        self.icca_lh = ICCA(channels)
        self.icca_hl = ICCA(channels)
        self.icca_hh = ICCA(channels)
        self.ccca = CCCA(channels)
        self.freq_gate = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, channels * 4),
        )
        if self.freeze_freq_gate:
            self.const_freq_gate_logits = nn.Parameter(torch.zeros(channels * 4))
        self.text_k = nn.Linear(text_dim, channels)
        self.text_v = nn.Linear(text_dim, channels)
        self.vis_q = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.spatial_gate_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.GELU(),
        )
        self.attn = SpatialSelfAttention2D(channels * 4) if use_attention else nn.Identity()
        self.branch_scale = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, 2 * channels),
        )
        nn.init.zeros_(self.branch_scale[-1].weight)
        nn.init.zeros_(self.branch_scale[-1].bias)

    def _token_grounding(self, agg: torch.Tensor, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        b, _, h, w = agg.shape
        q = self.vis_q(agg).flatten(2).transpose(1, 2)
        k = self.text_k(text_tokens)
        v = self.text_v(text_tokens)
        attn = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.bool().unsqueeze(1), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        grounded = torch.bmm(attn, v).transpose(1, 2).reshape(b, self.channels, h, w)
        spatial = torch.sigmoid(self.spatial_gate_proj(grounded))
        return spatial

    def forward(self, x: torch.Tensor, text_pooled: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        b, c, h, w = x.shape
        f0 = self.local_conv(x)
        ll, lh, hl, hh = haar_dwt2d(f0)
        ll = self.icca_ll(ll)
        lh = self.icca_lh(lh)
        hl = self.icca_hl(hl)
        hh = self.icca_hh(hh)
        if self.freeze_freq_gate:
            gates = self.const_freq_gate_logits.unsqueeze(0).expand(b, -1)
        else:
            gates = self.freq_gate(text_pooled)
        g_ll, g_lh, g_hl, g_hh = torch.chunk(gates, 4, dim=1)
        a_ll = torch.sigmoid(g_ll).view(b, c, 1, 1)
        a_lh = torch.sigmoid(g_lh).view(b, c, 1, 1)
        a_hl = torch.sigmoid(g_hl).view(b, c, 1, 1)
        a_hh = torch.sigmoid(g_hh).view(b, c, 1, 1)
        ll = ll * a_ll
        lh = lh * a_lh * self.lh_hl_scale
        hl = hl * a_hl * self.lh_hl_scale
        hh = hh * a_hh
        if self.hh_drop_mode == "zero":
            hh = torch.zeros_like(hh)
            hh_scale_debug = torch.zeros(1, dtype=torch.float32)
        elif self.hh_drop_mode == "learned":
            hh_scale = torch.sigmoid(self.hh_scale_logit).to(dtype=hh.dtype)
            hh = hh * hh_scale
            hh_scale_debug = hh_scale.detach().mean().cpu().float().view(1)
        else:
            hh_scale_debug = torch.ones(1, dtype=torch.float32)
        ll, lh, hl, hh = self.ccca(ll, lh, hl, hh)
        agg = torch.cat([ll, lh, hl, hh], dim=1)
        if self.disable_spatial_mask:
            spatial_mask = agg.new_ones((b, 1, agg.shape[-2], agg.shape[-1]))
        else:
            spatial_mask = self._token_grounding(agg, text_tokens, text_mask)
            spatial_mask = spatial_mask.pow(self.spatial_sharpen_power)
        if self.capture_debug:
            self.last_debug = {
                "a_ll_mean": a_ll.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_lh_mean": a_lh.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_hl_mean": a_hl.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_hh_mean": a_hh.mean(dim=(1, 2, 3)).detach().cpu(),
                "lh_hl_scale": torch.tensor([self.lh_hl_scale], dtype=torch.float32),
                "hh_scale": hh_scale_debug,
                "spatial_mask": spatial_mask.detach().cpu(),
            }
        agg = agg * spatial_mask
        agg = self.mixer(agg)
        agg = self.attn(agg)
        ll, lh, hl, hh = torch.chunk(agg, 4, dim=1)
        rec = haar_idwt2d(ll, lh, hl, hh)
        scales = self.branch_scale(text_pooled)
        beta, gamma = torch.chunk(scales, 2, dim=1)
        beta = (1.0 + 0.1 * torch.tanh(beta)).view(b, c, 1, 1)
        gamma = (0.5 + 0.5 * torch.sigmoid(gamma)).view(b, c, 1, 1)
        out = beta * f0 + gamma * rec
        return out


class EncoderStageText(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        text_dim: int,
        use_attention: bool = False,
        drop_hh: bool = False,
        hh_drop_mode: str | None = None,
        lh_hl_scale: float = 1.0,
        spatial_sharpen_power: float = 2.0,
        freeze_freq_gate: bool = False,
        disable_spatial_mask: bool = False,
    ) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.tgfs = TGFSBlockV2(
            out_channels,
            text_dim=text_dim,
            use_attention=use_attention,
            drop_hh=drop_hh,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=lh_hl_scale,
            spatial_sharpen_power=spatial_sharpen_power,
            freeze_freq_gate=freeze_freq_gate,
            disable_spatial_mask=disable_spatial_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        text_pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.block(x)
        return self.tgfs(x, text_pooled=text_pooled, text_tokens=text_tokens, text_mask=text_mask)


class TGFSDecoderStageV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        text_dim: int,
        use_attention: bool,
        drop_hh: bool = False,
        hh_drop_mode: str | None = None,
        lh_hl_scale: float = 1.0,
        spatial_sharpen_power: float = 2.0,
        freeze_freq_gate: bool = False,
        disable_spatial_mask: bool = False,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)
        self.tgfs = TGFSBlockV2(
            out_channels,
            text_dim=text_dim,
            use_attention=use_attention,
            drop_hh=drop_hh,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=lh_hl_scale,
            spatial_sharpen_power=spatial_sharpen_power,
            freeze_freq_gate=freeze_freq_gate,
            disable_spatial_mask=disable_spatial_mask,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, text_pooled: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.tgfs(x, text_pooled=text_pooled, text_tokens=text_tokens, text_mask=text_mask)
        return x


class PlainDecoderStageV2(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)
        self.freqa = FreqA(out_channels, use_attention=use_attention)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        text_pooled: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del text_pooled, text_tokens, text_mask
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.freqa(x)
        return x


class LFAENetTGFSv2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        bottleneck_channels: int = 768,
        text_dim: int = 256,
        vocab_size: int = 30522,
        text_encoder_type: str = 'simple',
        text_backbone_path: str = 'BiomedVLP-CXR-BERT-specialized',
        freeze_text_backbone: bool = True,
        use_external_text_encoder: bool = False,
        drop_hh_in_decoder: bool = True,
        hh_drop_mode: str | None = None,
        low_level_hf_scale: float = 0.6,
        spatial_sharpen_power: float = 2.0,
        use_deep_supervision: bool = False,
        fusion_mode: str = "decoder",
        unfreeze_last_n: int = 0,
        lora_r: int = 0,
        freeze_freq_gate: bool = False,
        disable_spatial_mask: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        if fusion_mode not in {"encoder", "decoder", "both"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if hh_drop_mode is None:
            hh_drop_mode = "zero" if drop_hh_in_decoder else "keep"
        if hh_drop_mode not in {"zero", "keep", "learned"}:
            raise ValueError(f"Unsupported hh_drop_mode: {hh_drop_mode}")
        self.hh_drop_mode = hh_drop_mode
        self.fusion_mode = fusion_mode
        self.encoder_uses_text = fusion_mode in {"encoder", "both"}
        self.decoder_uses_text = fusion_mode in {"decoder", "both"}
        self.use_external_text_encoder = use_external_text_encoder
        self.text_encoder_type = text_encoder_type
        if not use_external_text_encoder:
            if text_encoder_type == 'simple':
                self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, text_dim=text_dim)
            elif text_encoder_type in {'biomedvlp-cxr-bert', 'cxr-bert'}:
                self.text_encoder = CXRBertTextEncoder(
                    model_dir=text_backbone_path,
                    out_dim=text_dim,
                    freeze=freeze_text_backbone,
                    use_mean_pool=True,
                    unfreeze_last_n=unfreeze_last_n,
                    lora_r=lora_r,
                )
            else:
                raise ValueError(f'Unsupported text_encoder_type: {text_encoder_type}')
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        encoder_stage_cls = EncoderStageText if self.encoder_uses_text else EncoderStagePlain
        decoder_stage_cls = TGFSDecoderStageV2 if self.decoder_uses_text else PlainDecoderStageV2
        if self.encoder_uses_text:
            self.enc1 = encoder_stage_cls(
                c1,
                c1,
                text_dim=text_dim,
                use_attention=False,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.enc2 = encoder_stage_cls(
                c1,
                c2,
                text_dim=text_dim,
                use_attention=False,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.enc3 = encoder_stage_cls(
                c2,
                c3,
                text_dim=text_dim,
                use_attention=False,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=1.0,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.enc4 = encoder_stage_cls(
                c3,
                c4,
                text_dim=text_dim,
                use_attention=True,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=1.0,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.bottleneck = encoder_stage_cls(
                c4,
                bottleneck_channels,
                text_dim=text_dim,
                use_attention=True,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=1.0,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
        else:
            self.enc1 = encoder_stage_cls(c1, c1, use_attention=False)
            self.enc2 = encoder_stage_cls(c1, c2, use_attention=False)
            self.enc3 = encoder_stage_cls(c2, c3, use_attention=False)
            self.enc4 = encoder_stage_cls(c3, c4, use_attention=True)
            self.bottleneck = encoder_stage_cls(c4, bottleneck_channels, use_attention=True)
        if self.decoder_uses_text:
            self.dec4 = decoder_stage_cls(
                bottleneck_channels,
                c4,
                c4,
                text_dim=text_dim,
                use_attention=True,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=1.0,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.dec3 = decoder_stage_cls(
                c4,
                c3,
                c3,
                text_dim=text_dim,
                use_attention=True,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=1.0,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.dec2 = decoder_stage_cls(
                c3,
                c2,
                c2,
                text_dim=text_dim,
                use_attention=False,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
            self.dec1 = decoder_stage_cls(
                c2,
                c1,
                c1,
                text_dim=text_dim,
                use_attention=False,
                drop_hh=drop_hh_in_decoder,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                freeze_freq_gate=freeze_freq_gate,
                disable_spatial_mask=disable_spatial_mask,
            )
        else:
            self.dec4 = decoder_stage_cls(bottleneck_channels, c4, c4, use_attention=True)
            self.dec3 = decoder_stage_cls(c4, c3, c3, use_attention=True)
            self.dec2 = decoder_stage_cls(c3, c2, c2, use_attention=False)
            self.dec1 = decoder_stage_cls(c2, c1, c1, use_attention=False)
        self.final_refine = ConvBlock(c1, c1)
        self.use_deep_supervision = use_deep_supervision
        if use_deep_supervision:
            self.aux_head_d4 = nn.Conv2d(c4, num_classes, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(c3, num_classes, kernel_size=1)
            self.aux_head_d2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def set_debug_capture(self, enabled: bool = True) -> None:
        for stage in (self.dec4, self.dec3, self.dec2, self.dec1):
            if hasattr(stage, "tgfs"):
                stage.tgfs.capture_debug = enabled

    def get_debug_outputs(self) -> dict[str, Optional[dict[str, torch.Tensor]]]:
        return {
            "dec4": getattr(getattr(self.dec4, "tgfs", None), "last_debug", None),
            "dec3": getattr(getattr(self.dec3, "tgfs", None), "last_debug", None),
            "dec2": getattr(getattr(self.dec2, "tgfs", None), "last_debug", None),
            "dec1": getattr(getattr(self.dec1, "tgfs", None), "last_debug", None),
        }

    def _encode_text(self, token_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], text_features: Optional[torch.Tensor]):
        if text_features is not None:
            pooled = text_features
            tokens = pooled.unsqueeze(1)
            return tokens, pooled
        if token_ids is None:
            raise ValueError('Either token_ids or text_features must be provided.')
        if self.use_external_text_encoder:
            raise ValueError('Model is configured with use_external_text_encoder=True, but text_features was not provided.')
        return self.text_encoder(token_ids, attention_mask)

    def forward(
        self,
        image: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        text_tokens, text_pooled = self._encode_text(token_ids=token_ids, attention_mask=attention_mask, text_features=text_features)
        x0 = self.stem(image)
        e1 = self.enc1(x0, text_pooled, text_tokens, attention_mask)
        x1 = self.pool(e1)
        e2 = self.enc2(x1, text_pooled, text_tokens, attention_mask)
        x2 = self.pool(e2)
        e3 = self.enc3(x2, text_pooled, text_tokens, attention_mask)
        x3 = self.pool(e3)
        e4 = self.enc4(x3, text_pooled, text_tokens, attention_mask)
        x4 = self.pool(e4)
        b = self.bottleneck(x4, text_pooled, text_tokens, attention_mask)
        d4 = self.dec4(b, e4, text_pooled, text_tokens, attention_mask)
        d3 = self.dec3(d4, e3, text_pooled, text_tokens, attention_mask)
        d2 = self.dec2(d3, e2, text_pooled, text_tokens, attention_mask)
        d1 = self.dec1(d2, e1, text_pooled, text_tokens, attention_mask)
        d1 = self.final_refine(d1)
        logits = self.head(d1)
        if return_aux and self.use_deep_supervision:
            aux = {
                "d4": F.interpolate(self.aux_head_d4(d4), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                "d3": F.interpolate(self.aux_head_d3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False),
                "d2": F.interpolate(self.aux_head_d2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False),
            }
            return logits, aux
        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 1, 224, 224, device=device)
    toks = torch.randint(1, 1000, (2, 24), device=device)
    attn = toks.ne(0).long()
    model = LFAENetTGFSv2(in_channels=1, num_classes=1, text_dim=256, vocab_size=2000, text_encoder_type='simple', use_external_text_encoder=False, drop_hh_in_decoder=True).to(device)
    y = model(x, token_ids=toks, attention_mask=attn)
    print('Output:', tuple(y.shape))
