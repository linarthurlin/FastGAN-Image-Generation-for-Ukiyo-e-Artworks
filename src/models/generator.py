"""FastGAN Generator (Liu et al., 2021).

Key components:
- InitLayer          : noise → 4×4 feature map  (ConvTranspose2d + BN + GLU)
- UpBlock            : ×2 upsample + Conv + BN + GLU
- UpBlockComp        : same but with an extra Conv for higher-resolution stages
- SLEBlock           : Skip-Layer channel-wise Excitation
                       connects an early-stage feature map to a later one via
                       AdaptiveAvgPool → Conv → Sigmoid modulation
- Generator          : wires the above into a progressive synthesis network
                       with SLE skip connections every other stage

Channel schedule  nfc(sz) = min(512, nfc_base × 256 // sz)
  nfc_base=16, image_size=256 → {4:512, 8:512, 16:256, 32:128, 64:64, 128:32, 256:16}
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nfc(sz: int, nfc_base: int = 16, max_ch: int = 512) -> int:
    """Return channel count for a feature map at spatial resolution `sz`."""
    return min(max_ch, nfc_base * (256 // sz))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class InitLayer(nn.Module):
    """Expand noise vector to 4×4 feature map."""
    def __init__(self, z_dim: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, out_ch * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(dim=1),   # (B, out_ch*2, 4, 4) → (B, out_ch, 4, 4)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.net(z)


class UpBlock(nn.Module):
    """×2 upsample with single conv, BN and GLU activation."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlockComp(nn.Module):
    """×2 upsample with two convs (deeper variant for high-res stages)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(dim=1),
            nn.Conv2d(out_ch, out_ch * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SLEBlock(nn.Module):
    """Skip-Layer channel-wise Excitation.

    low-res feature (any spatial size) → AdaptiveAvgPool(4) → Conv 4→1 → Sigmoid
    → channel-wise multiplicative gate applied to high-res feature.
    """
    def __init__(self, ch_low: int, ch_high: int):
        super().__init__()
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.SiLU(),
            nn.Conv2d(ch_low, ch_high, 4, 1, 0, bias=False),  # (B, ch_high, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * self.excite(low)  # broadcast over H × W


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """FastGAN Generator.

    Supports image_size ∈ {64, 128, 256}.

    Args:
        z_dim:      Latent noise dimension.
        nfc_base:   Channel multiplier; nfc(sz) = min(512, nfc_base × 256//sz).
        image_size: Output resolution (64 / 128 / 256).
        img_ch:     Output image channels (default 3).
    """

    def __init__(
        self,
        z_dim: int = 256,
        nfc_base: int = 16,
        image_size: int = 256,
        img_ch: int = 3,
    ):
        super().__init__()
        assert image_size in (64, 128, 256), "image_size must be 64, 128 or 256"
        self.image_size = image_size

        def ch(sz):
            return _nfc(sz, nfc_base)

        # ---- Initial layer  4×4 ----------------------------------------
        self.init = InitLayer(z_dim, ch(4))

        # ---- Upsampling stages ------------------------------------------
        #   4 → 8 → 16 → 32 → 64 [→ 128 → 256]
        self.up_8   = UpBlock(ch(4),  ch(8))
        self.up_16  = UpBlock(ch(8),  ch(16))
        self.up_32  = UpBlockComp(ch(16), ch(32))
        self.up_64  = UpBlockComp(ch(32), ch(64))

        if image_size >= 128:
            self.up_128 = UpBlockComp(ch(64),  ch(128))
        if image_size >= 256:
            self.up_256 = UpBlockComp(ch(128), ch(256))

        # ---- SLE skip connections ---------------------------------------
        # Pattern: feat at resolution R skips to the stage that outputs R*8
        # 64px  : feat_8  → skip → scale feat_64
        # 128px : feat_8  → feat_64,  feat_16 → feat_128
        # 256px : feat_8  → feat_64,  feat_16 → feat_128, feat_32 → feat_256
        self.sle_64  = SLEBlock(ch(8),  ch(64))
        if image_size >= 128:
            self.sle_128 = SLEBlock(ch(16), ch(128))
        if image_size >= 256:
            self.sle_256 = SLEBlock(ch(32), ch(256))

        # ---- Output layer -----------------------------------------------
        out_ch = ch(image_size)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(out_ch, img_ch, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feat_4  = self.init(z)
        feat_8  = self.up_8(feat_4)
        feat_16 = self.up_16(feat_8)
        feat_32 = self.up_32(feat_16)
        feat_64 = self.sle_64(feat_8, self.up_64(feat_32))

        if self.image_size == 64:
            return self.to_rgb(feat_64)

        feat_128 = self.sle_128(feat_16, self.up_128(feat_64))
        if self.image_size == 128:
            return self.to_rgb(feat_128)

        feat_256 = self.sle_256(feat_32, self.up_256(feat_128))
        return self.to_rgb(feat_256)