import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.generator import _nfc


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlockComp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class SimpleDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        mid = max(in_ch // 4, 32)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),          # 8  → 16
            nn.Conv2d(in_ch, mid * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid * 2),
            nn.GLU(dim=1),                                         # → mid
            nn.Upsample(scale_factor=2, mode='nearest'),          # 16 → 32
            nn.Conv2d(mid, mid * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid * 2),
            nn.GLU(dim=1),                                         # → mid
            nn.Conv2d(mid, out_ch, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):

    def __init__(
        self,
        nfc_base: int = 16,
        image_size: int = 256,
        img_ch: int = 3,
    ):
        super().__init__()
        assert image_size in (64, 128, 256), "image_size must be 64, 128 or 256"

        def ch(sz):
            return _nfc(sz, nfc_base)

        # ---- from-RGB (no BN on first layer, paper convention) ----------
        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, ch(image_size), 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---- Encoder stages ---------------------------------------------
        #  image_size → ... → 8 → 4
        stages = []
        sz = image_size
        while sz > 8:
            sz_next = sz // 2
            stages.append(DownBlockComp(ch(sz), ch(sz_next)))
            sz = sz_next
        # sz is now 8×8
        self.encoder = nn.Sequential(*stages)

        # Final down 8→4
        self.down_4 = DownBlock(ch(8), ch(4))

        # ---- Classifier -------------------------------------------------
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch(4) * 4 * 4, 1),
        )

        # ---- Decoder (from 8×8 features → crop reconstruction) ---------
        self.decoder = SimpleDecoder(ch(8), img_ch)

        # ---- Part-crop branch (classifies a random crop at native res) --
        # Implemented as a lightweight 2-layer head on a 32×32 crop
        crop_sz = image_size // 4
        crop_ch = ch(crop_sz)
        self.part_from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, crop_ch, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.part_down = nn.Sequential(
            DownBlock(crop_ch, ch(crop_sz // 2)),
            DownBlock(ch(crop_sz // 2), ch(crop_sz // 4)),
        )
        self.part_classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(ch(crop_sz // 4) * 16, 1),
        )

    # ------------------------------------------------------------------
    def _random_crop(self, x: torch.Tensor, crop_sz: int) -> torch.Tensor:
        B, C, H, W = x.shape
        if H <= crop_sz:
            return F.interpolate(x, size=(crop_sz, crop_sz),
                                 mode='bilinear', align_corners=False)
        top  = torch.randint(0, H - crop_sz + 1, (1,)).item()
        left = torch.randint(0, W - crop_sz + 1, (1,)).item()
        return x[:, :, top:top + crop_sz, left:left + crop_sz]

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        crop_sz = self.image_size if hasattr(self, 'image_size') else x.size(-1)
        crop_sz = x.size(-1) // 4

        # Main encoder path
        feat = self.from_rgb(x)
        feat_8 = self.encoder(feat)          # (B, ch(8), 8, 8)
        feat_4 = self.down_4(feat_8)         # (B, ch(4), 4, 4)

        logit = self.classify(feat_4)        # (B, 1)

        # Decoder reconstruction (used only during D training)
        recon = self.decoder(feat_8)         # (B, 3, 32/64/128, *)

        # Part-crop discriminator
        crop  = self._random_crop(x, crop_sz)
        p_feat = self.part_from_rgb(crop)
        p_feat = self.part_down(p_feat)
        part_logit = self.part_classify(p_feat)  # (B, 1)

        return logit, part_logit, recon