from __future__ import annotations

from .lfaenet_tgfs_v2_oldtext import restore_old_tgfs_text_logic
from .lfaenet_tgfs_v3 import LFAENetTGFSv3


class LFAENetTGFSv3OldText(LFAENetTGFSv3):
    """TGFS-v3 with the old BERT/text conditioning path.

    This keeps v3 options such as ResNet-50 visual encoder and decoder-side TGFS,
    while restoring the old text behavior:
    - BERT token outputs are mean-pooled;
    - pooled text is fed directly to TGFS frequency gates / branch scales.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["text_pooling"] = "mean"
        super().__init__(*args, **kwargs)
        self.old_text_logic = True
        self.old_text_gate_norm_replaced = restore_old_tgfs_text_logic(self)
