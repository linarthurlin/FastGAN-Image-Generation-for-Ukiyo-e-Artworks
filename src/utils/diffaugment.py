import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual augmentation functions
# All operate on (B, C, H, W) tensors in [-1, 1].
# ---------------------------------------------------------------------------

def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    shift = torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5   # (-0.5, 0.5)
    return x + shift


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)                              # luminance proxy
    scale  = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2    # (0, 2)
    return (x - x_mean) * scale + x_mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    scale  = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5  # (0.5, 1.5)
    return (x - x_mean) * scale + x_mean


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    B, C, H, W = x.shape
    sh = int(H * ratio + 0.5)
    sw = int(W * ratio + 0.5)
    ty = torch.randint(-sh, sh + 1, (B,), device=x.device)  # (B,)
    tx = torch.randint(-sw, sw + 1, (B,), device=x.device)

    # Build per-image shifted sampling grids  (B, H, W)
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device),
                             torch.arange(W, device=x.device), indexing="ij")
    gy = (gy.unsqueeze(0) + ty.view(B, 1, 1)).clamp(0, H - 1)  # (B, H, W)
    gx = (gx.unsqueeze(0) + tx.view(B, 1, 1)).clamp(0, W - 1)
    gb = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, H, W)

    # x[gb, :, gy, gx]: PyTorch places the slice dim last → (B, H, W, C)
    # permute back to (B, C, H, W)
    return x[gb, :, gy, gx].permute(0, 3, 1, 2).contiguous()


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    B, C, H, W = x.shape
    ch = int(H * ratio + 0.5)
    cw = int(W * ratio + 0.5)

    oy = torch.randint(0, H - ch + 1, (B,), device=x.device)  # top-left y
    ox = torch.randint(0, W - cw + 1, (B,), device=x.device)  # top-left x

    mask = torch.ones_like(x)
    for i in range(B):
        mask[i, :, oy[i]:oy[i] + ch, ox[i]:ox[i] + cw] = 0.0
    return x * mask


# ---------------------------------------------------------------------------
# Policy dispatcher
# ---------------------------------------------------------------------------

_POLICY_MAP = {
    "color":       [_rand_brightness, _rand_saturation, _rand_contrast],
    "translation": [_rand_translation],
    "cutout":      [_rand_cutout],
}


def diffaugment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    if not policy:
        return x

    for p in policy.split(","):
        p = p.strip()
        for fn in _POLICY_MAP.get(p, []):
            x = fn(x)

    return x
