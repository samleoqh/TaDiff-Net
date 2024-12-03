"""
Evaluation metrics for TaDiff model.
Handles metric computation and evaluation of model predictions.
"""
from typing import Tuple, Dict, Union, Optional

import torch
import numpy as np
from torchmetrics import (
    Dice,
    MeanAbsoluteError as MAE,
    # StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR
)

from src.evaluation.ssim import SSIM

def setup_metrics(device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Initialize all evaluation metrics.
    
    Args:
        device: Device to place the metrics on
        
    Returns:
        Dictionary containing initialized metrics
    """
    return {
        'dice_th05': Dice(threshold=0.5).to(device),
        'dice_th025': Dice(threshold=0.25).to(device),
        'dice_th075': Dice(threshold=0.75).to(device),
        'mae': MAE().to(device),
        'psnr': PSNR().to(device),
        'ssim': SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=255,
            size_average=True,
            channel=3
        ).to(device)
    }

def calculate_relative_volume_difference(
    prediction: Union[torch.Tensor, np.ndarray],
    truth: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Tuple[float, int]:
    """
    Calculate relative absolute volume difference.
    
    Args:
        prediction: Predicted mask
        truth: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Tuple of (relative volume difference, predicted volume)
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().detach().numpy()
        
    pred_binary = (prediction >= threshold).astype(bool)
    truth_binary = truth.astype(bool)
    
    pred_volume = np.count_nonzero(pred_binary) + 1
    truth_volume = np.count_nonzero(truth_binary) + 1
    
    relative_diff = (pred_volume - truth_volume) / float(truth_volume)
    
    return relative_diff, pred_volume - 1

def calculate_metrics(
    metrics: Dict[str, torch.nn.Module],
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reference: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate all metrics for a prediction.
    
    Args:
        metrics: Dictionary of metric functions
        prediction: Model prediction
        ground_truth: Ground truth data
        reference: Reference data (optional)
        
    Returns:
        Dictionary containing computed metrics
    """
    results = {}
    
    # Normalize predictions and ground truth
    pred_norm = ((prediction.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
    gt_norm = ((ground_truth.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
    
    # Calculate basic metrics
    results['ssim'] = metrics['ssim'](pred_norm, gt_norm).cpu().detach().numpy()
    results['mae'] = metrics['mae'](prediction.clamp(-1, 1), 
                                  ground_truth.clamp(-1, 1)).cpu().detach().numpy()
    results['psnr'] = metrics['psnr'](pred_norm, gt_norm).cpu().detach().numpy()
    
    # Calculate DICE scores at different thresholds
    results['dice25'] = metrics['dice_th025'](prediction, ground_truth).cpu().detach().numpy()
    results['dice50'] = metrics['dice_th05'](prediction, ground_truth).cpu().detach().numpy()
    results['dice75'] = metrics['dice_th075'](prediction, ground_truth).cpu().detach().numpy()
    
    # Calculate volume differences at different thresholds
    results['ravd25'], vol25 = calculate_relative_volume_difference(prediction, ground_truth, 0.25)
    results['ravd50'], vol50 = calculate_relative_volume_difference(prediction, ground_truth, 0.50)
    results['ravd75'], vol75 = calculate_relative_volume_difference(prediction, ground_truth, 0.75)
    
    results['pred_vol'] = vol50 / 100.0  # Convert to appropriate units
    
    # Calculate reference-based metrics if reference is provided
    if reference is not None:
        results['dice_ref'] = metrics['dice_th05'](prediction, reference).cpu().detach().numpy()
        results['ravd_ref'], _ = calculate_relative_volume_difference(prediction, reference, 0.50)
    
    # Round all results to 3 decimal places
    results = {k: np.round(v, 3) for k, v in results.items()}
    
    return results

def calculate_tumor_volumes(
    labels: torch.Tensor,
    axis: Tuple[int, ...] = (1, 2, 3)
) -> torch.Tensor:
    """
    Calculate tumor volumes from label masks.
    
    Args:
        labels: Label tensor
        axis: Axes to sum over for volume calculation
        
    Returns:
        Tensor containing calculated volumes
    """
    return torch.sum(labels, dim=axis).float()

def get_slice_indices(
    volume_tensor: torch.Tensor,
    top_k: int,
    min_volume: Optional[float] = None
) -> torch.Tensor:
    """
    Get indices of top-k slices by volume.
    
    Args:
        volume_tensor: Tensor containing volumes
        top_k: Number of top slices to return
        min_volume: Minimum volume threshold
        
    Returns:
        Tensor containing selected slice indices
    """
    if min_volume is not None:
        volume_tensor = torch.where(
            volume_tensor < min_volume,
            torch.zeros_like(volume_tensor),
            volume_tensor
        )
    
    _, indices = torch.topk(volume_tensor, top_k)
    return indices
