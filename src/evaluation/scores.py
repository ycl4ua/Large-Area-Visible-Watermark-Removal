"""Metrics used by the LaMa-style validation evaluator.

The paper reports PSNR, SSIM, RMSE, RMSEw and LPIPS.  FID and the segmentation
aware metrics from LaMa are not used here and have been removed.
"""
from abc import abstractmethod, ABC
from math import log10

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lpips import PerceptualLoss
from .ssim import SSIM


def get_groupings(groups):
    """Map group labels to indices (used for mask-area grouping)."""
    label_groups, count_groups = np.unique(groups, return_counts=True)
    indices = np.argsort(groups)

    grouping = dict()
    cur_start = 0
    for label, count in zip(label_groups, count_groups):
        cur_end = cur_start + count
        grouping[label] = indices[cur_start:cur_end]
        cur_start = cur_end
    return grouping


class EvaluatorScore(nn.Module):
    @abstractmethod
    def forward(self, pred_batch, target_batch, mask):
        pass

    @abstractmethod
    def get_value(self, groups=None, states=None):
        pass

    @abstractmethod
    def reset(self):
        pass


class PairwiseScore(EvaluatorScore, ABC):
    def __init__(self):
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None \
            else self.individual_values

        total_results = {
            'mean': individual_values.mean(),
            'std': individual_values.std()
        }

        if groups is None:
            return total_results, None

        group_results = dict()
        for label, index in get_groupings(groups).items():
            group_scores = individual_values[index]
            group_results[label] = {
                'mean': group_scores.mean(),
                'std': group_scores.std()
            }
        return total_results, group_results

    def reset(self):
        self.individual_values = []


class SSIMScore(PairwiseScore):
    def __init__(self, window_size=11):
        super().__init__()
        self.score = SSIM(window_size=window_size, size_average=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch)
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values


class PSNR(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, pred_batch, target_batch):
        assert pred_batch.shape == target_batch.shape
        items = []
        for pred, target in zip(pred_batch, target_batch):
            items.append(10 * log10(1 / F.mse_loss(pred, target).item()))
        return torch.tensor(items, device=self.device)


class PSNRScore(PairwiseScore):
    def __init__(self, device):
        super().__init__()
        self.score = PSNR(device).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch)
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values


class LPIPSScore(PairwiseScore):
    def __init__(self, model='net-lin', net='vgg', model_path=None, use_gpu=True):
        super().__init__()
        self.score = PerceptualLoss(model=model, net=net, model_path=model_path,
                                    use_gpu=use_gpu, spatial=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch).flatten()
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values
