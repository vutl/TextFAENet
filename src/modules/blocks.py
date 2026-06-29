from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import HaarDWT2D, HaarIDWT2D


FREQ_BANDS = {"ll", "lh", "hl", "hh"}


def normalize_freq_drop_bands(drop_bands: str | None) -> frozenset[str]:
    if drop_bands is None:
        return frozenset()
    if drop_bands.strip().lower() in {"", "none", "keep"}:
        return frozenset()
    bands = {x for x in re.split(r"[,;+\s]+", drop_bands.strip().lower()) if x}
    unknown = bands - FREQ_BANDS
    if unknown:
        raise ValueError(f"Unsupported frequency bands to drop: {sorted(unknown)}")
    return frozenset(bands)


def apply_freq_drop(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    drop_bands: frozenset[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not drop_bands:
        return ll, lh, hl, hh
    if "ll" in drop_bands:
        ll = torch.zeros_like(ll)
    if "lh" in drop_bands:
        lh = torch.zeros_like(lh)
    if "hl" in drop_bands:
        hl = torch.zeros_like(hl)
    if "hh" in drop_bands:
        hh = torch.zeros_like(hh)
    return ll, lh, hl, hh


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.act(out + identity)
        return out


class ICCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.pool(x).view(b, c)
        z = F.relu(self.fc1(z), inplace=True)
        a = torch.sigmoid(self.fc2(z)).view(b, c, 1, 1)
        return x * a


class CCCA(nn.Module):
    """Cross-component refinement with channel-wise interactions across bands."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        alpha = torch.full((4, 4), -2.0)
        alpha.fill_diagonal_(0.0)
        self.alpha = nn.Parameter(alpha)
        self.norm = nn.BatchNorm2d(channels)

    @staticmethod
    def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_desc = F.adaptive_avg_pool2d(a, 1).flatten(1)
        b_desc = F.adaptive_avg_pool2d(b, 1).flatten(1)
        a_desc = F.normalize(a_desc, p=2, dim=1)
        b_desc = F.normalize(b_desc, p=2, dim=1)
        return a_desc * b_desc  # [B, C]

    def forward(
        self,
        fll: torch.Tensor,
        flh: torch.Tensor,
        fhl: torch.Tensor,
        fhh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bands = [fll, flh, fhl, fhh]
        gates = torch.sigmoid(self.alpha)
        outs: list[torch.Tensor] = []

        for i in range(4):
            out_i = bands[i]
            for j in range(4):
                if i == j:
                    continue
                sim = self._channel_similarity(bands[i], bands[j]).unsqueeze(-1).unsqueeze(-1)
                out_i = out_i + gates[i, j] * sim * bands[j]

            out_i = self.norm(out_i)
            outs.append(out_i)

        return outs[0], outs[1], outs[2], outs[3]


class FrequencyMixer(nn.Module):
    def __init__(self, channels: int, use_attention: bool = True, num_heads: int = 4) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.channels = channels
        if use_attention:
            if channels % num_heads != 0:
                num_heads = 1
            self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(channels)
            self.ffn = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.GELU(),
                nn.Linear(channels * 2, channels),
            )
            self.norm2 = nn.LayerNorm(channels)
        else:
            self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_attention:
            y = self.dw(x)
            y = self.pw(y)
            y = self.act(self.bn(y))
            return x + y

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class FreqA(nn.Module):
    """DWT -> ICCA -> CCCA -> mixer -> iDWT with residual wrapper."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_attention: bool = True,
        attn_heads: int = 4,
        freq_drop_bands: str | None = None,
    ) -> None:
        super().__init__()
        self.freq_drop_bands = normalize_freq_drop_bands(freq_drop_bands)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.icca_ll = ICCA(channels, reduction)
        self.icca_lh = ICCA(channels, reduction)
        self.icca_hl = ICCA(channels, reduction)
        self.icca_hh = ICCA(channels, reduction)

        self.ccca = CCCA(channels)
        self.mixer = FrequencyMixer(4 * channels, use_attention=use_attention, num_heads=attn_heads)
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(x)
        ll, lh, hl, hh = apply_freq_drop(ll, lh, hl, hh, self.freq_drop_bands)
        ll = self.icca_ll(ll)
        lh = self.icca_lh(lh)
        hl = self.icca_hl(hl)
        hh = self.icca_hh(hh)

        ll, lh, hl, hh = self.ccca(ll, lh, hl, hh)
        ll, lh, hl, hh = apply_freq_drop(ll, lh, hl, hh, self.freq_drop_bands)
        f_agg = torch.cat([ll, lh, hl, hh], dim=1)
        f_mix = self.mixer(f_agg)
        ll, lh, hl, hh = torch.chunk(f_mix, 4, dim=1)

        y = self.idwt(ll, lh, hl, hh)
        y = self.out_norm(y)
        return x + y


class TGFSBlock(nn.Module):
    """Text-Guided Frequency Selection block for decoder stages."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        reduction: int = 16,
        use_attention: bool = False,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.icca_ll = ICCA(channels, reduction)
        self.icca_lh = ICCA(channels, reduction)
        self.icca_hl = ICCA(channels, reduction)
        self.icca_hh = ICCA(channels, reduction)

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, 4 * channels),
        )

        self.ccca = CCCA(channels)
        self.mixer = FrequencyMixer(4 * channels, use_attention=use_attention, num_heads=attn_heads)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        f0 = self.local(x)
        ll, lh, hl, hh = self.dwt(f0)

        ll = self.icca_ll(ll)
        lh = self.icca_lh(lh)
        hl = self.icca_hl(hl)
        hh = self.icca_hh(hh)

        gates = self.text_mlp(text_vec)  # [B, 4C]
        g_ll, g_lh, g_hl, g_hh = torch.chunk(gates, 4, dim=1)

        a_ll = torch.sigmoid(g_ll).unsqueeze(-1).unsqueeze(-1)
        a_lh = torch.sigmoid(g_lh).unsqueeze(-1).unsqueeze(-1)
        a_hl = torch.sigmoid(g_hl).unsqueeze(-1).unsqueeze(-1)
        a_hh = torch.sigmoid(g_hh).unsqueeze(-1).unsqueeze(-1)

        ll = ll * a_ll
        lh = lh * a_lh
        hl = hl * a_hl
        hh = hh * a_hh

        ll, lh, hl, hh = self.ccca(ll, lh, hl, hh)
        f_agg = torch.cat([ll, lh, hl, hh], dim=1)
        f_mix = self.mixer(f_agg)

        ll, lh, hl, hh = torch.chunk(f_mix, 4, dim=1)
        f_rec = self.idwt(ll, lh, hl, hh)

        out = self.out_bn(self.out_conv(f0 + f_rec))
        return x + out
