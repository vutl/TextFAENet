from __future__ import annotations

import torch.nn as nn

from .lfaenet_tgfs_v2 import LFAENetTGFSv2


def restore_old_tgfs_text_logic(module: nn.Module) -> int:
    """Restore the pre-attention-pooling TGFS text-gating path.

    Current TGFS normalizes pooled text before frequency gates and branch scales.
    The old logic fed the pooled text vector directly into those MLPs. This helper
    replaces every TGFS block gate-normalization layer by identity.
    """
    replaced = 0
    for child in module.modules():
        if hasattr(child, "text_gate_norm"):
            child.text_gate_norm = nn.Identity()
            replaced += 1
    return replaced


class LFAENetTGFSv2OldText(LFAENetTGFSv2):
    """TGFS-v2 with the old BERT/text conditioning path.

    This keeps the v2 visual/frequency architecture intact but disables the newer
    text changes:
    - no learned attention pooling for external BERT text encoders;
    - no LayerNorm before frequency gates / branch scales.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["text_pooling"] = "mean"
        super().__init__(*args, **kwargs)
        self.old_text_logic = True
        self.old_text_gate_norm_replaced = restore_old_tgfs_text_logic(self)
