import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from typing import Tuple, Dict


def combined_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    l1_weight: float = 1.0,
    mse_weight: float = 1.0,
    ssim_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculates a weighted combination of L1, MSE, and SSIM losses.

    Args:
        pred (torch.Tensor): The predicted image tensor of shape [B, C, H, W].
        gt (torch.Tensor): The ground truth image tensor of shape [B, C, H, W].
        l1_weight (float): The weight for the L1 loss component.
        mse_weight (float): The weight for the MSE loss component.
        ssim_weight (float): The weight for the SSIM loss component.

    Returns:
        A tuple containing:
        - torch.Tensor: The total weighted loss.
        - Dict[str, float]: A dictionary containing the values of individual losses for logging.
    """
    log_dict = {}

    # L1 Loss
    l1_loss_val = F.l1_loss(pred, gt)
    log_dict["l1_loss"] = l1_loss_val.item()

    # MSE Loss
    mse_loss_val = F.mse_loss(pred, gt)
    log_dict["mse_loss"] = mse_loss_val.item()

    # SSIM Loss (note: SSIM is a similarity metric, so loss is 1 - ssim)
    data_range = 1.0 # Assuming data is normalized to [0, 1]
    ssim_value = MF.structural_similarity_index_measure(pred, gt, data_range=data_range)
    ssim_loss = 1 - ssim_value
    log_dict["ssim_loss"] = ssim_loss.item()
    log_dict["ssim_value"] = ssim_value.item()

    # Total Weighted Loss
    total_loss = (
        l1_weight * l1_loss_val
        + mse_weight * mse_loss_val
        + ssim_weight * ssim_loss
    )
    
    log_dict["total_loss"] = total_loss.item()

    return total_loss, log_dict
