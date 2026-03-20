import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse

    _HAS_PYTORCH_WAVELETS = True
except Exception:
    DWTForward = None
    DWTInverse = None
    _HAS_PYTORCH_WAVELETS = False


class HaarDWT2D(nn.Module):
    """Channel-wise 2D Haar DWT implemented with grouped conv2d."""

    def __init__(self, backend: str = "auto", wave: str = "db1", mode: str = "zero") -> None:
        super().__init__()
        self.backend = backend
        if self.backend == "auto":
            self.backend = "pytorch_wavelets" if _HAS_PYTORCH_WAVELETS else "manual"

        if self.backend == "pytorch_wavelets":
            if not _HAS_PYTORCH_WAVELETS:
                raise RuntimeError("pytorch_wavelets backend requested but package is not installed")
            self.dwt = DWTForward(J=1, wave=wave, mode=mode)
            self.filt = None
            return

        if self.backend != "manual":
            raise ValueError(f"Unsupported backend: {self.backend}")

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "pytorch_wavelets":
            yl, yh_list = self.dwt(x)
            yh = yh_list[0]  # [B, C, 3, H/2, W/2]
            lh = yh[:, :, 0, :, :]
            hl = yh[:, :, 1, :, :]
            hh = yh[:, :, 2, :, :]
            return yl, lh, hl, hh

        _, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"H and W must be even for HaarDWT2D, got {(h, w)}")

        weight = self.filt.repeat(c, 1, 1, 1)  # [4C, 1, 2, 2]
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # [B, 4C, H/2, W/2]
        ll, lh, hl, hh = torch.chunk(y, 4, dim=1)
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """Inverse channel-wise 2D Haar DWT with grouped conv_transpose2d."""

    def __init__(self, backend: str = "auto", wave: str = "db1", mode: str = "zero") -> None:
        super().__init__()
        self.backend = backend
        if self.backend == "auto":
            self.backend = "pytorch_wavelets" if _HAS_PYTORCH_WAVELETS else "manual"

        if self.backend == "pytorch_wavelets":
            if not _HAS_PYTORCH_WAVELETS:
                raise RuntimeError("pytorch_wavelets backend requested but package is not installed")
            self.idwt = DWTInverse(wave=wave, mode=mode)
            self.filt = None
            return

        if self.backend != "manual":
            raise ValueError(f"Unsupported backend: {self.backend}")

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        for t in (lh, hl, hh):
            if t.shape != ll.shape:
                raise ValueError("All sub-bands must have the same shape for HaarIDWT2D")

        if self.backend == "pytorch_wavelets":
            yh = torch.stack([lh, hl, hh], dim=2)  # [B, C, 3, H/2, W/2]
            return self.idwt((ll, [yh]))

        _, c, _, _ = ll.shape
        x = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4C, H/2, W/2]
        weight = self.filt.repeat(c, 1, 1, 1)  # [4C, 1, 2, 2]
        y = F.conv_transpose2d(x, weight, stride=2, padding=0, groups=c)
        return y
