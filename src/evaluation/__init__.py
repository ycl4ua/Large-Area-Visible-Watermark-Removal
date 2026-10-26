import logging

import torch

from src.evaluation.evaluator import InpaintingEvaluatorOnline
from src.evaluation.scores import PSNRScore, SSIMScore, LPIPSScore


def make_evaluator(kind='default', ssim=True, lpips=True, integral_kind=None, **kwargs):
    """Build the validation evaluator.

    FID is not used in the paper and is not implemented here.  The evaluator
    reports SSIM, LPIPS and PSNR (RMSE/RMSEw are computed by ``bin/test.py``).
    """
    logging.info(f'Make evaluator {kind}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = {}

    if ssim:
        metrics['ssim'] = SSIMScore()
    if lpips:
        metrics['lpips'] = LPIPSScore(use_gpu=torch.cuda.is_available())
    metrics['psnr'] = PSNRScore(device)

    if integral_kind is not None and integral_kind != 'None':
        raise ValueError('integral_kind is not supported; use None')

    if kind == 'default':
        return InpaintingEvaluatorOnline(scores=metrics, integral_func=None,
                                         integral_title=None, **kwargs)
    raise ValueError(f'Unknown evaluator kind {kind}')
