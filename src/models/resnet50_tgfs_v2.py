from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfaenet_tgfs_v2 import (
    CXRBertTextEncoder,
    ConvBlock,
    PlainDecoderStageV2,
    SimpleTextEncoder,
    TGFSDecoderStageV2,
    normalize_freq_drop_bands,
)


def _resnet50_weights(pretrained: str):
    pretrained = pretrained.lower()
    if pretrained in {"none", "false", "0", ""}:
        return None
    if pretrained in {"imagenet", "default", "imagenet1k_v2"}:
        from torchvision.models import ResNet50_Weights

        return ResNet50_Weights.IMAGENET1K_V2
    if pretrained == "imagenet1k_v1":
        from torchvision.models import ResNet50_Weights

        return ResNet50_Weights.IMAGENET1K_V1
    raise ValueError(f"Unsupported ResNet-50 pretrained setting: {pretrained}")


def _adapt_resnet_input_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1 and conv.weight.shape[1] == 3:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            repeat = (in_channels + conv.weight.shape[1] - 1) // conv.weight.shape[1]
            w = conv.weight.repeat(1, repeat, 1, 1)[:, :in_channels]
            w = w * (conv.weight.shape[1] / float(in_channels))
            new_conv.weight.copy_(w)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature pyramid for 224x224 segmentation inputs.

    Outputs:
    - e1: 1/2 resolution, 64 channels
    - e2: 1/4 resolution, 256 channels
    - e3: 1/8 resolution, 512 channels
    - e4: 1/16 resolution, 1024 channels
    - b:  1/32 resolution, 2048 channels
    """

    def __init__(self, in_channels: int = 1, pretrained: str = "imagenet") -> None:
        super().__init__()
        from torchvision.models import resnet50

        weights = _resnet50_weights(pretrained)
        net = resnet50(weights=weights)
        net.conv1 = _adapt_resnet_input_channels(net.conv1, in_channels)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.stem(x)
        x = self.maxpool(e1)
        e2 = self.layer1(x)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        b = self.layer4(e4)
        return e1, e2, e3, e4, b


class ResNet50FAENet(nn.Module):
    """ImageNet ResNet-50 visual encoder with a frequency-aware decoder, no text."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        visual_pretrained: str = "imagenet",
        freq_drop_bands: str | None = None,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = decoder_channels
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=visual_pretrained)
        self.dec4 = PlainDecoderStageV2(2048, 1024, c4, use_attention=True, freq_drop_bands=freq_drop_bands)
        self.dec3 = PlainDecoderStageV2(c4, 512, c3, use_attention=True, freq_drop_bands=freq_drop_bands)
        self.dec2 = PlainDecoderStageV2(c3, 256, c2, use_attention=False, freq_drop_bands=freq_drop_bands)
        self.dec1 = PlainDecoderStageV2(c2, 64, c1, use_attention=False, freq_drop_bands=freq_drop_bands)
        self.final_refine = ConvBlock(c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out_size = image.shape[-2:]
        e1, e2, e3, e4, b = self.encoder(image)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.head(self.final_refine(d1))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class ResNet50TGFSv2(nn.Module):
    """ImageNet ResNet-50 visual encoder with the TGFS-v2 decoder.

    This keeps the text-guided frequency-selection decoder intact while replacing the
    from-scratch visual trunk with a stronger ResNet-50 feature pyramid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        text_dim: int = 256,
        vocab_size: int = 30522,
        text_encoder_type: str = "simple",
        text_backbone_path: str = "BiomedVLP-CXR-BERT-specialized",
        freeze_text_backbone: bool = True,
        visual_pretrained: str = "imagenet",
        hh_drop_mode: str = "keep",
        low_level_hf_scale: float = 0.6,
        spatial_sharpen_power: float = 2.0,
        use_deep_supervision: bool = False,
        fusion_mode: str = "decoder",
        unfreeze_last_n: int = 0,
        lora_r: int = 0,
        freq_drop_bands: str | None = None,
        text_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if fusion_mode != "decoder":
            raise ValueError(
                "ResNet50TGFSv2 currently supports decoder-side TGFS only. "
                "Use the from-scratch LFAENetTGFSv2 for encoder/both text fusion."
            )
        if hh_drop_mode not in {"zero", "keep", "learned"}:
            raise ValueError(f"Unsupported hh_drop_mode: {hh_drop_mode}")
        self.freq_drop_bands = normalize_freq_drop_bands(freq_drop_bands)
        self.fusion_mode = fusion_mode
        self.decoder_uses_text = True
        self.use_deep_supervision = use_deep_supervision
        c1, c2, c3, c4 = decoder_channels

        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=visual_pretrained)
        if text_encoder_type == "simple":
            self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, text_dim=text_dim)
        elif text_encoder_type in {"biomedvlp-cxr-bert", "cxr-bert"}:
            self.text_encoder = CXRBertTextEncoder(
                model_dir=text_backbone_path,
                out_dim=text_dim,
                freeze=freeze_text_backbone,
                use_mean_pool=True,
                unfreeze_last_n=unfreeze_last_n,
                lora_r=lora_r,
                pooling=text_pooling,
            )
        else:
            raise ValueError(f"Unsupported text_encoder_type: {text_encoder_type}")

        decoder_stage_cls = TGFSDecoderStageV2
        self.dec4 = decoder_stage_cls(
            2048,
            1024,
            c4,
            text_dim=text_dim,
            use_attention=True,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=1.0,
            spatial_sharpen_power=spatial_sharpen_power,
            freq_drop_bands=freq_drop_bands,
        )
        self.dec3 = decoder_stage_cls(
            c4,
            512,
            c3,
            text_dim=text_dim,
            use_attention=True,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=1.0,
            spatial_sharpen_power=spatial_sharpen_power,
            freq_drop_bands=freq_drop_bands,
        )
        self.dec2 = decoder_stage_cls(
            c3,
            256,
            c2,
            text_dim=text_dim,
            use_attention=False,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=low_level_hf_scale,
            spatial_sharpen_power=spatial_sharpen_power,
            freq_drop_bands=freq_drop_bands,
        )
        self.dec1 = decoder_stage_cls(
            c2,
            64,
            c1,
            text_dim=text_dim,
            use_attention=False,
            hh_drop_mode=hh_drop_mode,
            lh_hl_scale=low_level_hf_scale,
            spatial_sharpen_power=spatial_sharpen_power,
            freq_drop_bands=freq_drop_bands,
        )
        self.final_refine = ConvBlock(c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        if use_deep_supervision:
            self.aux_head_d4 = nn.Conv2d(c4, num_classes, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(c3, num_classes, kernel_size=1)
            self.aux_head_d2 = nn.Conv2d(c2, num_classes, kernel_size=1)

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

    def _encode_text(
        self,
        token_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_features is not None:
            pooled = text_features
            return pooled.unsqueeze(1), pooled
        if token_ids is None:
            raise ValueError("Either token_ids or text_features must be provided.")
        return self.text_encoder(token_ids, attention_mask)

    def forward(
        self,
        image: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out_size = image.shape[-2:]
        text_tokens, text_pooled = self._encode_text(token_ids, attention_mask, text_features)
        e1, e2, e3, e4, b = self.encoder(image)
        d4 = self.dec4(b, e4, text_pooled, text_tokens, attention_mask)
        d3 = self.dec3(d4, e3, text_pooled, text_tokens, attention_mask)
        d2 = self.dec2(d3, e2, text_pooled, text_tokens, attention_mask)
        d1 = self.dec1(d2, e1, text_pooled, text_tokens, attention_mask)
        d1 = self.final_refine(d1)
        logits = self.head(d1)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        if return_aux and self.use_deep_supervision:
            aux = {
                "d4": F.interpolate(self.aux_head_d4(d4), size=out_size, mode="bilinear", align_corners=False),
                "d3": F.interpolate(self.aux_head_d3(d3), size=out_size, mode="bilinear", align_corners=False),
                "d2": F.interpolate(self.aux_head_d2(d2), size=out_size, mode="bilinear", align_corners=False),
            }
            return logits, aux
        return logits
