"""Evaluation metrics used in the paper.

The reported RMSE/RMSEw are on the 0-255 scale (the original implementation
multiplies the [0, 1] RMSE by 256).
"""
import torch
import torch.nn.functional as F

from src.evaluation.ssim import SSIM


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item() * 256.0


def compute_rmse_masked(pred, target, mask):
    """RMSE averaged inside the mask (mask=1)."""
    mask = mask.expand_as(pred)
    denom = mask.sum().clamp_min(1e-6)
    mse = ((pred - target) ** 2 * mask).sum() / denom
    return torch.sqrt(mse).item() * 256.0


class SSIMMetric:
    def __init__(self, device):
        self.impl = SSIM(size_average=False).to(device)

    @torch.no_grad()
    def __call__(self, pred, target):
        return self.impl(pred, target).mean().item()


class LPIPSMetric:
    """Vendored LPIPS (VGG) using models/lpips_models/vgg.pth."""

    def __init__(self, device):
        from src.evaluation.lpips import PerceptualLoss

        self.impl = PerceptualLoss(model='net-lin', net='vgg', use_gpu=device.type == 'cuda').to(device)
        self.impl.eval()

    @torch.no_grad()
    def __call__(self, pred, target):
        # PerceptualLoss expects [0, 1] inputs and normalizes them internally.
        return self.impl(pred, target).mean().item()
