import torch
import torch.nn.functional as F


def d_logistic_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    real_loss = F.softplus(-real_logit).mean()   # -log σ(D(real))
    fake_loss = F.softplus(fake_logit).mean()    # -log(1 - σ(D(fake))) = softplus(D(fake))
    return real_loss + fake_loss


def g_logistic_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logit).mean()        # -log σ(D(fake))


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape[-1] != recon.shape[-1] or target.shape[-2] != recon.shape[-2]:
        target = F.interpolate(
            target, size=recon.shape[-2:], mode='bilinear', align_corners=False
        )
    return F.l1_loss(recon, target)
