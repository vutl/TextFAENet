from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.blocks import ConvBlock, FreqA, TGFSBlock


class SimpleTextEncoder(nn.Module):
    """Minimal text encoder for bootstrapping training/inference pipelines."""

    def __init__(self, vocab_size: int, text_dim: int, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=pad_id)
        self.gru = nn.GRU(text_dim, text_dim, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(text_dim, text_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # token_ids: [B, L]
        if attention_mask is None:
            mask = token_ids.ne(self.pad_id).float().unsqueeze(-1)
        else:
            mask = attention_mask.float().unsqueeze(-1)

        x = self.embedding(token_ids)
        x, _ = self.gru(x)
        x = self.proj(x)

        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled


class CXRBertTextEncoder(nn.Module):
    """Wrapper for local BiomedVLP-CXR-BERT-specialized checkpoint."""

    def __init__(
        self,
        model_dir: str,
        out_dim: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        from transformers import AutoConfig, AutoModel

        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Text backbone directory not found: {model_dir}")

        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        hidden = int(getattr(cfg, "hidden_size", 768))
        self.proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = token_ids.ne(0).long()

        if hasattr(self.model, "bert"):
            outputs = self.model.bert(
                input_ids=token_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
        else:
            outputs = self.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state

        pooled = last_hidden[:, 0, :]
        return self.proj(pooled)


class TGFSDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        text_dim: int,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)
        self.tgfs = TGFSBlock(out_channels, text_dim=text_dim, use_attention=use_attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.tgfs(x, text_vec)
        return x


class LFAENetTGFS(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        bottleneck_channels: int = 768,
        text_dim: int = 256,
        vocab_size: int = 30522,
        text_encoder_type: str = "simple",
        text_backbone_path: str = "BiomedVLP-CXR-BERT-specialized",
        freeze_text_backbone: bool = True,
        use_external_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        self.use_external_text_encoder = use_external_text_encoder
        self.text_encoder_type = text_encoder_type
        if not use_external_text_encoder:
            if text_encoder_type == "simple":
                self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, text_dim=text_dim)
            elif text_encoder_type in {"biomedvlp-cxr-bert", "cxr-bert"}:
                self.text_encoder = CXRBertTextEncoder(
                    model_dir=text_backbone_path,
                    out_dim=text_dim,
                    freeze=freeze_text_backbone,
                )
            else:
                raise ValueError(f"Unsupported text_encoder_type: {text_encoder_type}")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = nn.Sequential(ConvBlock(c1, c1), FreqA(c1, use_attention=False))

        self.enc2 = nn.Sequential(ConvBlock(c1, c2), FreqA(c2, use_attention=False))

        self.enc3 = nn.Sequential(ConvBlock(c2, c3), FreqA(c3, use_attention=False))

        self.enc4 = nn.Sequential(ConvBlock(c3, c4), FreqA(c4, use_attention=True))

        self.bottleneck = nn.Sequential(
            ConvBlock(c4, bottleneck_channels),
            FreqA(bottleneck_channels, use_attention=True),
        )

        self.dec4 = TGFSDecoderStage(bottleneck_channels, c4, c4, text_dim=text_dim, use_attention=True)
        self.dec3 = TGFSDecoderStage(c4, c3, c3, text_dim=text_dim, use_attention=True)
        self.dec2 = TGFSDecoderStage(c3, c2, c2, text_dim=text_dim, use_attention=False)
        self.dec1 = TGFSDecoderStage(c2, c1, c1, text_dim=text_dim, use_attention=False)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def _encode_text(
        self,
        token_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        text_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_features is not None:
            return text_features

        if token_ids is None:
            raise ValueError("Either token_ids or text_features must be provided.")

        if self.use_external_text_encoder:
            raise ValueError(
                "Model is configured with use_external_text_encoder=True, but text_features was not provided."
            )

        return self.text_encoder(token_ids, attention_mask)

    def forward(
        self,
        image: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_vec = self._encode_text(token_ids=token_ids, attention_mask=attention_mask, text_features=text_features)

        x0 = self.stem(image)
        e1 = self.enc1(x0)

        x1 = self.pool(e1)
        e2 = self.enc2(x1)

        x2 = self.pool(e2)
        e3 = self.enc3(x2)

        x3 = self.pool(e3)
        e4 = self.enc4(x3)

        x4 = self.pool(e4)
        b = self.bottleneck(x4)

        d4 = self.dec4(b, e4, text_vec)
        d3 = self.dec3(d4, e3, text_vec)
        d2 = self.dec2(d3, e2, text_vec)
        d1 = self.dec1(d2, e1, text_vec)

        logits = self.head(d1)
        return logits
