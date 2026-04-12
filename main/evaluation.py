import torch
import torch.nn.functional as F
from typing import Tuple


def cross_correlation(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-correlation (CC) between predicted and ground truth images.

    Args:
        pred (torch.Tensor): The predicted image tensor of shape [B, C, H, W].
        gt (torch.Tensor): The ground truth image tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor: The mean cross-correlation value.
    """
    b, n_spectral, h, w = pred.shape
    pred_reshaped = pred.view(b, n_spectral, -1)
    gt_reshaped = gt.view(b, n_spectral, -1)

    mean_pred = torch.mean(pred_reshaped, dim=2, keepdim=True)
    mean_gt = torch.mean(gt_reshaped, dim=2, keepdim=True)

    numerator = torch.sum((pred_reshaped - mean_pred) * (gt_reshaped - mean_gt), dim=2)
    denominator = torch.sqrt(
        torch.sum((pred_reshaped - mean_pred) ** 2, dim=2) *
        torch.sum((gt_reshaped - mean_gt) ** 2, dim=2)
    )

    cc = numerator / (denominator + 1e-8)
    return torch.mean(cc)
