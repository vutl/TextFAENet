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


def make_norm(channels: int, norm_type: str = "bn", num_groups: int = 8) -> nn.Module:
    if norm_type == "gn":
        groups = max(1, min(num_groups, channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "bn",
        depth: int = 2,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if depth not in {2, 3}:
            raise ValueError(f"ConvBlock depth must be 2 or 3, got {depth}")
        self.depth = depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = make_norm(out_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = make_norm(out_channels, norm_type)
        if depth == 3:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn3 = make_norm(out_channels, norm_type)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels, norm_type),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        if self.depth == 3:
            x = self.act(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return self.dropout(x)


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
        q = self.q(x).flatten(2).transpose(1, 2).contiguous()
        k = self.k(x).flatten(2).contiguous()
        v = self.v(x).flatten(2).transpose(1, 2).contiguous()
        attn = torch.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).contiguous().reshape(b, c, h, w)
        out = self.proj(out)
        return x + self.gamma * out


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_a = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + delta * self.scaling


def apply_lora_to_backbone(module: nn.Module, r: int, alpha: float) -> int:
    target_keys = ("query", "key", "value", "q_proj", "k_proj", "v_proj")
    count = 0

    def _walk(parent: nn.Module) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            lname = name.lower()
            if isinstance(child, nn.Linear) and any(k in lname for k in target_keys):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha))
                count += 1
            else:
                _walk(child)

    _walk(module)
    return count


class TextFiLM2D(nn.Module):
    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        beta, gamma = torch.chunk(self.proj(text_vec), 2, dim=1)
        scale = (1.0 + 0.1 * torch.tanh(gamma)).reshape(b, c, 1, 1)
        shift = (0.1 * torch.tanh(beta)).reshape(b, c, 1, 1)
        return x * scale + shift


class SpatialTextFusion(nn.Module):
    def __init__(self, channels: int, text_dim: int, norm_type: str = "bn") -> None:
        super().__init__()
        self.channels = channels
        self.text_k = nn.Linear(text_dim, channels)
        self.text_v = nn.Linear(text_dim, channels)
        self.vis_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.spatial_gate_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels, norm_type),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.spatial_gate_proj[-1].weight)
        nn.init.zeros_(self.spatial_gate_proj[-1].bias)

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.vis_q(x).flatten(2).transpose(1, 2).contiguous()
        k = self.text_k(text_tokens)
        v = self.text_v(text_tokens)
        
        attn = torch.bmm(q, k.transpose(1, 2).contiguous()) / (self.channels ** 0.5)
        if text_mask is not None:
            attn = attn.masked_fill(~text_mask.bool().unsqueeze(1), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        
        grounded = torch.bmm(attn, v).transpose(1, 2).contiguous().reshape(b, self.channels, h, w)
        gate = self.spatial_gate_proj(grounded)
        return x * (1.0 + 0.1 * torch.tanh(gate))


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
    # MPS-compatible: use stack+permute+reshape instead of scatter assignment.
    # Equivalent to placing x00,x01,x10,x11 at even/odd pixel positions.
    b, c, h, w = ll.shape
    x00 = (ll + lh + hl + hh) * 0.5   # even row, even col
    x01 = (ll - lh + hl - hh) * 0.5   # even row, odd  col
    x10 = (ll + lh - hl - hh) * 0.5   # odd  row, even col
    x11 = (ll - lh - hl + hh) * 0.5   # odd  row, odd  col
    # Stack along a new dim → (b, c, h, w, 4) then rearrange to (b, c, 2h, 2w)
    # row-interleave: stack x0_, x1_ along axis=3 → (b,c,h,2,w) → ...
    top = torch.stack([x00, x01], dim=4)   # (b, c, h, w, 2)  → even row: [col0 col1]
    bot = torch.stack([x10, x11], dim=4)   # (b, c, h, w, 2)  → odd  row
    # top/bot: (b, c, h, w, 2) → reshape to (b, c, h, 2w)
    top = top.reshape(b, c, h, w * 2)
    bot = bot.reshape(b, c, h, w * 2)
    # interleave rows: stack along dim=3 → (b,c,h,2,2w) → (b,c,2h,2w)
    out = torch.stack([top, bot], dim=3).reshape(b, c, h * 2, w * 2)
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
        a = torch.sigmoid(self.mlp(z)).reshape(b, c, 1, 1)
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
        w_ll, w_lh, w_hl, w_hh = [t.contiguous() for t in torch.chunk(w, 4, dim=1)]
        ll = ll * w_ll.reshape(b, c, 1, 1)
        lh = lh * w_lh.reshape(b, c, 1, 1)
        hl = hl * w_hl.reshape(b, c, 1, 1)
        hh = hh * w_hh.reshape(b, c, 1, 1)
        return ll, lh, hl, hh


class FreqA(nn.Module):
    def __init__(
        self,
        channels: int,
        use_attention: bool = False,
        norm_type: str = "bn",
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.icca_ll = ICCA(channels)
        self.icca_lh = ICCA(channels)
        self.icca_hl = ICCA(channels)
        self.icca_hh = ICCA(channels)
        self.ccca = CCCA(channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
            make_norm(channels * 4, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=1, bias=False),
            make_norm(channels * 4, norm_type),
            nn.ReLU(inplace=True),
        )
        self.attn = SpatialSelfAttention2D(channels * 4) if use_attention else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

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
        ll, lh, hl, hh = [t.contiguous() for t in torch.chunk(agg, 4, dim=1)]
        rec = haar_idwt2d(ll, lh, hl, hh)
        return x + self.dropout(rec)


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


class CXRBertTextEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str,
        out_dim: int,
        freeze: bool = True,
        use_mean_pool: bool = True,
        unfreeze_last_n: int = 0,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
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
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        if unfreeze_last_n > 0:
            self._unfreeze_last_n_layers(unfreeze_last_n)
        if lora_r > 0:
            wrapped = apply_lora_to_backbone(self.model, r=lora_r, alpha=lora_alpha)
            if wrapped == 0:
                raise RuntimeError("LoRA requested but no attention projection layers were found in text backbone")

    def _unfreeze_last_n_layers(self, unfreeze_last_n: int) -> None:
        target = self.model.bert if hasattr(self.model, "bert") else self.model
        encoder = getattr(target, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None and len(layers) > 0:
            n = min(unfreeze_last_n, len(layers))
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            return

        # Fallback for uncommon backbone structures.
        params = list(target.parameters())
        n = min(unfreeze_last_n, len(params))
        for p in params[-n:]:
            p.requires_grad = True

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            attention_mask = token_ids.ne(0).long()
        if hasattr(self.model, "bert"):
            outputs = self.model.bert(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
        else:
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
        tokens = self.token_proj(outputs.last_hidden_state.contiguous())
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
        hh_drop_mode: str = "keep",
        lh_hl_scale: float = 1.0,
        learnable_lh_hl_scale: bool = False,
        spatial_sharpen_power: float = 2.0,
        learnable_spatial_sharpen: bool = False,
        norm_type: str = "bn",
        dropout_p: float = 0.0,
        grounding_n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        if hh_drop_mode not in {"zero", "keep", "learned"}:
            raise ValueError(f"Unsupported hh_drop_mode: {hh_drop_mode}")
        self.hh_drop_mode = hh_drop_mode
        self.learnable_lh_hl_scale = learnable_lh_hl_scale
        if learnable_lh_hl_scale:
            self.lh_hl_scale = nn.Parameter(torch.tensor(float(lh_hl_scale), dtype=torch.float32))
        else:
            self.lh_hl_scale = float(lh_hl_scale)
        self.learnable_spatial_sharpen = learnable_spatial_sharpen
        if learnable_spatial_sharpen:
            self.spatial_sharpen_power = nn.Parameter(torch.tensor(float(spatial_sharpen_power), dtype=torch.float32))
        else:
            self.spatial_sharpen_power = float(spatial_sharpen_power)
        if self.hh_drop_mode == "learned":
            # Start close to dropping HH, then let training recover if useful.
            self.hh_keep_logit = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        # Multi-head grounding: pick a valid n_heads that divides channels.
        n_heads = max(1, int(grounding_n_heads))
        while channels % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.grounding_n_heads = n_heads
        self.grounding_head_dim = channels // n_heads
        self.capture_debug = False
        self.last_debug: Optional[dict[str, torch.Tensor]] = None
        # Raw spatial mask saved each forward (before sharpening), used for grounding supervision loss.
        self.last_spatial_mask: Optional[torch.Tensor] = None
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels, norm_type),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels, norm_type),
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
        self.text_k = nn.Linear(text_dim, channels)
        self.text_v = nn.Linear(text_dim, channels)
        self.vis_q = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.spatial_gate_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels, norm_type),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
            make_norm(channels * 4, norm_type),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=1, bias=False),
            make_norm(channels * 4, norm_type),
            nn.GELU(),
        )
        self.attn = SpatialSelfAttention2D(channels * 4) if use_attention else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.branch_scale = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, 2 * channels),
        )
        nn.init.zeros_(self.branch_scale[-1].weight)
        nn.init.zeros_(self.branch_scale[-1].bias)

    def _token_grounding(self, agg: torch.Tensor, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        b, _, h, w = agg.shape
        n_heads = self.grounding_n_heads
        head_dim = self.grounding_head_dim
        # Visual query: (B, C, H, W) -> (B, n_heads, H*W, head_dim)
        q = self.vis_q(agg)
        q = q.view(b, n_heads, head_dim, h * w).permute(0, 1, 3, 2).contiguous()
        # Text key/value: (B, T, C) -> (B, n_heads, T, head_dim)
        t = text_tokens.shape[1]
        k = self.text_k(text_tokens).view(b, t, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = self.text_v(text_tokens).view(b, t, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        attn = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        no_valid_token = None
        if attention_mask is not None:
            valid_tokens = attention_mask.bool()
            no_valid_token = ~valid_tokens.any(dim=1)
            if no_valid_token.any():
                # Empty-prompt ablations intentionally remove text. Avoid
                # softmax over all -inf and use a neutral spatial mask instead.
                valid_tokens = valid_tokens.clone()
                valid_tokens[no_valid_token] = True
            mask_b = valid_tokens.view(b, 1, 1, t)
            attn = attn.masked_fill(~mask_b, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        grounded = torch.matmul(attn, v)  # (B, n_heads, H*W, head_dim)
        grounded = grounded.permute(0, 1, 3, 2).contiguous().view(b, self.channels, h, w)
        spatial = torch.sigmoid(self.spatial_gate_proj(grounded))
        if no_valid_token is not None and no_valid_token.any():
            spatial = spatial.clone()
            spatial[no_valid_token] = 1.0
        return spatial

    def forward(self, x: torch.Tensor, text_pooled: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        b, c, h, w = x.shape
        f0 = self.local_conv(x)
        ll, lh, hl, hh = haar_dwt2d(f0)
        ll = self.icca_ll(ll)
        lh = self.icca_lh(lh)
        hl = self.icca_hl(hl)
        hh = self.icca_hh(hh)
        gates = self.freq_gate(text_pooled)
        g_ll, g_lh, g_hl, g_hh = [t.contiguous() for t in torch.chunk(gates, 4, dim=1)]
        a_ll = torch.sigmoid(g_ll).reshape(b, c, 1, 1)
        a_lh = torch.sigmoid(g_lh).reshape(b, c, 1, 1)
        a_hl = torch.sigmoid(g_hl).reshape(b, c, 1, 1)
        a_hh = torch.sigmoid(g_hh).reshape(b, c, 1, 1)
        ll = ll * a_ll
        if self.learnable_lh_hl_scale:
            lh_hl_scale = self.lh_hl_scale.clamp(0.0, 2.0)
        else:
            lh_hl_scale = ll.new_tensor(self.lh_hl_scale)
        lh = lh * a_lh * lh_hl_scale
        hl = hl * a_hl * lh_hl_scale
        hh = hh * a_hh
        if self.hh_drop_mode == "zero":
            hh = torch.zeros_like(hh)
        elif self.hh_drop_mode == "learned":
            hh = hh * torch.sigmoid(self.hh_keep_logit)
        ll, lh, hl, hh = self.ccca(ll, lh, hl, hh)
        agg = torch.cat([ll, lh, hl, hh], dim=1)
        spatial_mask_raw = self._token_grounding(agg, text_tokens, text_mask)
        # Save the raw sigmoid mask (before sharpening) so the train loop can
        # supervise it directly with the ground-truth segmentation mask.
        self.last_spatial_mask = spatial_mask_raw
        if self.learnable_spatial_sharpen:
            sharpen_power = self.spatial_sharpen_power.clamp(0.5, 4.0)
        else:
            sharpen_power = agg.new_tensor(self.spatial_sharpen_power)
        spatial_mask = spatial_mask_raw.pow(sharpen_power)
        if self.capture_debug:
            self.last_debug = {
                "a_ll_mean": a_ll.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_lh_mean": a_lh.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_hl_mean": a_hl.mean(dim=(1, 2, 3)).detach().cpu(),
                "a_hh_mean": a_hh.mean(dim=(1, 2, 3)).detach().cpu(),
                "lh_hl_scale": lh_hl_scale.detach().view(1).cpu(),
                "hh_drop_mode": torch.tensor(
                    [0 if self.hh_drop_mode == "zero" else 1 if self.hh_drop_mode == "keep" else 2],
                    dtype=torch.float32,
                ),
                "spatial_sharpen_power": sharpen_power.detach().view(1).cpu(),
                "spatial_mask": spatial_mask.detach().cpu(),
            }
        agg = agg * spatial_mask
        agg = self.mixer(agg)
        agg = self.attn(agg)
        ll, lh, hl, hh = [t.contiguous() for t in torch.chunk(agg, 4, dim=1)]
        rec = haar_idwt2d(ll, lh, hl, hh)
        scales = self.branch_scale(text_pooled)
        beta, gamma = [t.contiguous() for t in torch.chunk(scales, 2, dim=1)]
        beta = (1.0 + 0.1 * torch.tanh(beta)).reshape(b, c, 1, 1)
        gamma = (0.5 + 0.5 * torch.sigmoid(gamma)).reshape(b, c, 1, 1)
        out = beta * f0 + gamma * rec
        return self.dropout(out)


class TGFSDecoderStageV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        text_dim: int,
        use_attention: bool,
        hh_drop_mode: str = "keep",
        lh_hl_scale: float = 1.0,
        learnable_lh_hl_scale: bool = False,
        spatial_sharpen_power: float = 2.0,
        learnable_spatial_sharpen: bool = False,
        norm_type: str = "bn",
        dropout_p: float = 0.0,
        grounding_n_heads: int = 1,
        conv_block_depth: int = 2,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            depth=conv_block_depth,
            dropout_p=dropout_p,
        )
        self.tgfs = TGFSBlockV2(
            out_channels,
            text_dim=text_dim,
            use_attention=use_attention,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=lh_hl_scale,
            learnable_lh_hl_scale=learnable_lh_hl_scale,
            spatial_sharpen_power=spatial_sharpen_power,
            learnable_spatial_sharpen=learnable_spatial_sharpen,
            norm_type=norm_type,
            dropout_p=dropout_p,
            grounding_n_heads=grounding_n_heads,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, text_pooled: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.tgfs(x, text_pooled=text_pooled, text_tokens=text_tokens, text_mask=text_mask)
        return x


class ResNet50ImageEncoder(nn.Module):
    """ResNet-50 backbone returning multi-scale features.

    Output features (for input H x W):
        s0: (B, 64,   H/2,  W/2)
        s1: (B, 256,  H/4,  W/4)
        s2: (B, 512,  H/8,  W/8)
        s3: (B, 1024, H/16, W/16)
        s4: (B, 2048, H/32, W/32)

    `conv1` is adapted in-place to accept `in_channels` (defaults to 1 for
    grayscale medical images) by averaging the original 3-channel ImageNet
    weights. When `freeze_bn=True` (recommended for small-batch fine-tuning)
    all BatchNorm layers are kept in eval mode with frozen affine params.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True, freeze_bn: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights
        except ImportError as e:
            raise RuntimeError("torchvision is required for ResNet50 encoder") from e
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        if in_channels != 3:
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    avg_w = backbone.conv1.weight.mean(dim=1, keepdim=True)
                    if in_channels == 1:
                        new_conv1.weight.copy_(avg_w)
                    else:
                        new_conv1.weight.copy_(avg_w.repeat(1, in_channels, 1, 1))
            backbone.conv1 = new_conv1
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.channels_out = [64, 256, 512, 1024, 2048]
        self.freeze_bn = bool(freeze_bn)
        if self.freeze_bn:
            self._set_bn_eval_and_freeze()

    def _set_bn_eval_and_freeze(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)
        s1 = self.layer1(self.maxpool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s0, s1, s2, s3, s4


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
        unfreeze_last_n: int = 0,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        use_external_text_encoder: bool = False,
        fusion_mode: str = 'decoder',
        drop_hh_in_decoder: bool = True,
        hh_drop_mode: str | None = None,
        low_level_hf_scale: float = 0.6,
        learnable_low_level_hf_scale: bool = False,
        spatial_sharpen_power: float = 2.0,
        learnable_spatial_sharpen: bool = False,
        use_deep_supervision: bool = False,
        encoder_text_fusion: str = 'film',
        norm_type: str = 'bn',
        conv_block_depth: int = 2,
        dropout_p: float = 0.0,
        grounding_n_heads: int = 1,
        encoder_type: str = 'from_scratch',
        pretrained_image_encoder: bool = True,
        freeze_encoder_bn: bool = True,
    ) -> None:
        super().__init__()
        if fusion_mode not in {'decoder', 'both'}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.encoder_text_fusion = encoder_text_fusion
        self.norm_type = norm_type
        self.conv_block_depth = conv_block_depth
        self.dropout_p = dropout_p
        self.grounding_n_heads = grounding_n_heads
        c1, c2, c3, c4 = channels
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
                    lora_alpha=lora_alpha,
                )
            else:
                raise ValueError(f'Unsupported text_encoder_type: {text_encoder_type}')
        if encoder_type not in {'from_scratch', 'resnet50'}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        if hh_drop_mode is None:
            hh_drop_mode = "zero" if drop_hh_in_decoder else "keep"
        _cb_kwargs = dict(norm_type=norm_type, depth=conv_block_depth, dropout_p=dropout_p)
        _fa_kwargs = dict(norm_type=norm_type, dropout_p=dropout_p)
        _dec_kwargs = dict(
            norm_type=norm_type,
            dropout_p=dropout_p,
            grounding_n_heads=grounding_n_heads,
            conv_block_depth=conv_block_depth,
        )

        if encoder_type == 'resnet50':
            # ResNet-50 pretrained encoder. Skip channels follow ResNet stages:
            # s0=64@H/2, s1=256@H/4, s2=512@H/8, s3=1024@H/16, s4=2048@H/32.
            # FreqA is applied on s1/s2/s3 only; s4 (2048ch) is reduced to
            # `bottleneck_channels` via ConvBlock + FreqA. We skip text fusion
            # on the highest-res stem feature s0 to save memory.
            self.image_encoder = ResNet50ImageEncoder(
                in_channels=in_channels,
                pretrained=pretrained_image_encoder,
                freeze_bn=freeze_encoder_bn,
            )
            r0, r1, r2, r3, r4 = self.image_encoder.channels_out
            self.freqa_r1 = FreqA(r1, use_attention=False, **_fa_kwargs)
            self.freqa_r2 = FreqA(r2, use_attention=False, **_fa_kwargs)
            self.freqa_r3 = FreqA(r3, use_attention=True, **_fa_kwargs)
            self.bottleneck = nn.Sequential(
                ConvBlock(r4, bottleneck_channels, **_cb_kwargs),
                FreqA(bottleneck_channels, use_attention=True, **_fa_kwargs),
            )
            if self.fusion_mode == 'both':
                if encoder_text_fusion == 'cross_attn':
                    def _enc_fuse(ch):
                        return SpatialTextFusion(ch, text_dim, norm_type=norm_type)
                else:
                    def _enc_fuse(ch):
                        return TextFiLM2D(ch, text_dim)
                self.r1_text = _enc_fuse(r1)
                self.r2_text = _enc_fuse(r2)
                self.r3_text = _enc_fuse(r3)
                self.bottleneck_text = _enc_fuse(bottleneck_channels)
            # Decoder: match current channel progression at decoder outputs.
            dec_out_c4, dec_out_c3, dec_out_c2, dec_out_c1 = 512, 256, 128, 64
            self._resnet_dec_channels = (dec_out_c4, dec_out_c3, dec_out_c2, dec_out_c1)
            self.dec4 = TGFSDecoderStageV2(
                bottleneck_channels, r3, dec_out_c4,
                text_dim=text_dim, use_attention=True,
                hh_drop_mode=hh_drop_mode, lh_hl_scale=1.0, learnable_lh_hl_scale=False,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec3 = TGFSDecoderStageV2(
                dec_out_c4, r2, dec_out_c3,
                text_dim=text_dim, use_attention=True,
                hh_drop_mode=hh_drop_mode, lh_hl_scale=1.0, learnable_lh_hl_scale=False,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec2 = TGFSDecoderStageV2(
                dec_out_c3, r1, dec_out_c2,
                text_dim=text_dim, use_attention=False,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                learnable_lh_hl_scale=learnable_low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec1 = TGFSDecoderStageV2(
                dec_out_c2, r0, dec_out_c1,
                text_dim=text_dim, use_attention=False,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                learnable_lh_hl_scale=learnable_low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.final_refine = ConvBlock(dec_out_c1, dec_out_c1, **_cb_kwargs)
            self.use_deep_supervision = use_deep_supervision
            if use_deep_supervision:
                self.aux_head_d4 = nn.Conv2d(dec_out_c4, num_classes, kernel_size=1)
                self.aux_head_d3 = nn.Conv2d(dec_out_c3, num_classes, kernel_size=1)
                self.aux_head_d2 = nn.Conv2d(dec_out_c2, num_classes, kernel_size=1)
            self.head = nn.Conv2d(dec_out_c1, num_classes, kernel_size=1)
        else:
            # Original from-scratch encoder/decoder path (unchanged).
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
                make_norm(c1, norm_type),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.enc1 = nn.Sequential(ConvBlock(c1, c1, **_cb_kwargs), FreqA(c1, use_attention=False, **_fa_kwargs))
            self.enc2 = nn.Sequential(ConvBlock(c1, c2, **_cb_kwargs), FreqA(c2, use_attention=False, **_fa_kwargs))
            self.enc3 = nn.Sequential(ConvBlock(c2, c3, **_cb_kwargs), FreqA(c3, use_attention=False, **_fa_kwargs))
            self.enc4 = nn.Sequential(ConvBlock(c3, c4, **_cb_kwargs), FreqA(c4, use_attention=True, **_fa_kwargs))
            self.bottleneck = nn.Sequential(
                ConvBlock(c4, bottleneck_channels, **_cb_kwargs),
                FreqA(bottleneck_channels, use_attention=True, **_fa_kwargs),
            )
            if self.fusion_mode == 'both':
                if encoder_text_fusion == 'cross_attn':
                    def _enc_fuse(ch):
                        return SpatialTextFusion(ch, text_dim, norm_type=norm_type)
                else:
                    def _enc_fuse(ch):
                        return TextFiLM2D(ch, text_dim)
                self.enc1_text = _enc_fuse(c1)
                self.enc2_text = _enc_fuse(c2)
                self.enc3_text = _enc_fuse(c3)
                self.enc4_text = _enc_fuse(c4)
                self.bottleneck_text = _enc_fuse(bottleneck_channels)
            self.dec4 = TGFSDecoderStageV2(
                bottleneck_channels, c4, c4,
                text_dim=text_dim, use_attention=True,
                hh_drop_mode=hh_drop_mode, lh_hl_scale=1.0, learnable_lh_hl_scale=False,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec3 = TGFSDecoderStageV2(
                c4, c3, c3,
                text_dim=text_dim, use_attention=True,
                hh_drop_mode=hh_drop_mode, lh_hl_scale=1.0, learnable_lh_hl_scale=False,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec2 = TGFSDecoderStageV2(
                c3, c2, c2,
                text_dim=text_dim, use_attention=False,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                learnable_lh_hl_scale=learnable_low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.dec1 = TGFSDecoderStageV2(
                c2, c1, c1,
                text_dim=text_dim, use_attention=False,
                hh_drop_mode=hh_drop_mode,
                lh_hl_scale=low_level_hf_scale,
                learnable_lh_hl_scale=learnable_low_level_hf_scale,
                spatial_sharpen_power=spatial_sharpen_power,
                learnable_spatial_sharpen=learnable_spatial_sharpen,
                **_dec_kwargs,
            )
            self.final_refine = ConvBlock(c1, c1, **_cb_kwargs)
            self.use_deep_supervision = use_deep_supervision
            if use_deep_supervision:
                self.aux_head_d4 = nn.Conv2d(c4, num_classes, kernel_size=1)
                self.aux_head_d3 = nn.Conv2d(c3, num_classes, kernel_size=1)
                self.aux_head_d2 = nn.Conv2d(c2, num_classes, kernel_size=1)
            self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def set_debug_capture(self, enabled: bool = True) -> None:
        self.dec4.tgfs.capture_debug = enabled
        self.dec3.tgfs.capture_debug = enabled
        self.dec2.tgfs.capture_debug = enabled
        self.dec1.tgfs.capture_debug = enabled

    def get_debug_outputs(self) -> dict[str, Optional[dict[str, torch.Tensor]]]:
        return {
            "dec4": self.dec4.tgfs.last_debug,
            "dec3": self.dec3.tgfs.last_debug,
            "dec2": self.dec2.tgfs.last_debug,
            "dec1": self.dec1.tgfs.last_debug,
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
        if self.encoder_type == 'resnet50':
            s0, s1, s2, s3, s4 = self.image_encoder(image)
            s1 = self.freqa_r1(s1)
            s2 = self.freqa_r2(s2)
            s3 = self.freqa_r3(s3)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    s1 = self.r1_text(s1, text_tokens, attention_mask)
                    s2 = self.r2_text(s2, text_tokens, attention_mask)
                    s3 = self.r3_text(s3, text_tokens, attention_mask)
                else:
                    s1 = self.r1_text(s1, text_pooled)
                    s2 = self.r2_text(s2, text_pooled)
                    s3 = self.r3_text(s3, text_pooled)
            b = self.bottleneck(s4)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    b = self.bottleneck_text(b, text_tokens, attention_mask)
                else:
                    b = self.bottleneck_text(b, text_pooled)
            d4 = self.dec4(b, s3, text_pooled, text_tokens, attention_mask)
            d3 = self.dec3(d4, s2, text_pooled, text_tokens, attention_mask)
            d2 = self.dec2(d3, s1, text_pooled, text_tokens, attention_mask)
            d1 = self.dec1(d2, s0, text_pooled, text_tokens, attention_mask)
            # ResNet50 stem reduces spatial by 2; upsample dec1 back to input res.
            if d1.shape[-2:] != image.shape[-2:]:
                d1 = F.interpolate(d1, size=image.shape[-2:], mode='bilinear', align_corners=False)
            d1 = self.final_refine(d1)
            logits = self.head(d1)
        else:
            x0 = self.stem(image)
            e1 = self.enc1(x0)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    e1 = self.enc1_text(e1, text_tokens, attention_mask)
                else:
                    e1 = self.enc1_text(e1, text_pooled)
            x1 = self.pool(e1)
            e2 = self.enc2(x1)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    e2 = self.enc2_text(e2, text_tokens, attention_mask)
                else:
                    e2 = self.enc2_text(e2, text_pooled)
            x2 = self.pool(e2)
            e3 = self.enc3(x2)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    e3 = self.enc3_text(e3, text_tokens, attention_mask)
                else:
                    e3 = self.enc3_text(e3, text_pooled)
            x3 = self.pool(e3)
            e4 = self.enc4(x3)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    e4 = self.enc4_text(e4, text_tokens, attention_mask)
                else:
                    e4 = self.enc4_text(e4, text_pooled)
            x4 = self.pool(e4)
            b = self.bottleneck(x4)
            if self.fusion_mode == 'both':
                if self.encoder_text_fusion == 'cross_attn':
                    b = self.bottleneck_text(b, text_tokens, attention_mask)
                else:
                    b = self.bottleneck_text(b, text_pooled)
            d4 = self.dec4(b, e4, text_pooled, text_tokens, attention_mask)
            d3 = self.dec3(d4, e3, text_pooled, text_tokens, attention_mask)
            d2 = self.dec2(d3, e2, text_pooled, text_tokens, attention_mask)
            d1 = self.dec1(d2, e1, text_pooled, text_tokens, attention_mask)
            d1 = self.final_refine(d1)
            logits = self.head(d1)
        if return_aux:
            aux: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
            if self.use_deep_supervision:
                aux["d4"] = F.interpolate(self.aux_head_d4(d4), size=logits.shape[-2:], mode="bilinear", align_corners=False)
                aux["d3"] = F.interpolate(self.aux_head_d3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False)
                aux["d2"] = F.interpolate(self.aux_head_d2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False)
            grounding: dict[str, torch.Tensor] = {}
            for name, stage in (("dec4", self.dec4), ("dec3", self.dec3), ("dec2", self.dec2), ("dec1", self.dec1)):
                m = stage.tgfs.last_spatial_mask
                if m is not None:
                    grounding[name] = m
            if grounding:
                aux["grounding"] = grounding
            if aux:
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
