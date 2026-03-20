import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import FAENet, LFAENetTGFS


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_rgb = torch.randn(2, 3, 256, 256, device=device)
    model_fae = FAENet(in_channels=3, num_classes=6).to(device)
    y_fae = model_fae(x_rgb)
    print("FAENet output:", tuple(y_fae.shape))

    x_med = torch.randn(2, 1, 224, 224, device=device)
    tokens = torch.randint(low=1, high=1000, size=(2, 24), device=device)
    model_lfae = LFAENetTGFS(
        in_channels=1,
        num_classes=1,
        text_dim=256,
        vocab_size=2000,
        text_encoder_type="simple",
        use_external_text_encoder=False,
    ).to(device)
    y_lfae = model_lfae(x_med, token_ids=tokens)
    print("LFAENetTGFS(simple) output:", tuple(y_lfae.shape))

    model_lfae_cxr = LFAENetTGFS(
        in_channels=1,
        num_classes=1,
        text_dim=256,
        text_encoder_type="biomedvlp-cxr-bert",
        text_backbone_path="BiomedVLP-CXR-BERT-specialized",
        freeze_text_backbone=True,
        use_external_text_encoder=False,
    ).to(device)
    attn = tokens.ne(0).long()
    y_lfae_cxr = model_lfae_cxr(x_med, token_ids=tokens, attention_mask=attn)
    print("LFAENetTGFS(cxr-bert) output:", tuple(y_lfae_cxr.shape))


if __name__ == "__main__":
    main()
