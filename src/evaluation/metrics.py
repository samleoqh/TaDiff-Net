"""Metrics for evaluating model predictions."""
import torch
from torchmetrics import Dice, MeanAbsoluteError, PeakSignalNoiseRatio
from src.evaluation.ssim import SSIM
import numpy as np

def setup_metrics(device):
    """Initialize all evaluation metrics."""
    return {
        'dice_th05': Dice(threshold=0.5).to(device),
        'dice_th025': Dice(threshold=0.25).to(device), 
        'dice_th075': Dice(threshold=0.75).to(device),
        'mae': MeanAbsoluteError().to(device),
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': SSIM(win_size=11, win_sigma=1.5, data_range=255, 
                     size_average=True, channel=3).to(device)
    }

def calculate_metrics(metrics, prediction, ground_truth):
    """Calculate all metrics between prediction and ground truth."""
    metrics_dict = {}
    
    # Normalize predictions to 0-255 range for SSIM/PSNR
    pred_255 = normalize_to_255(prediction)
    gt_255 = normalize_to_255(ground_truth)
    
    metrics_dict['ssim'] = metrics['ssim'](pred_255, gt_255).cpu().detach().numpy()
    metrics_dict['mae'] = metrics['mae'](prediction.clamp(-1, 1), 
                                       ground_truth.clamp(-1, 1)).cpu().detach().numpy()
    metrics_dict['psnr'] = metrics['psnr'](pred_255, gt_255).cpu().detach().numpy()
    
    return metrics_dict

def normalize_to_255(x):
    """Normalize tensor to 0-255 range."""
    return ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)

def calculate_tumor_volumes(labels):
    """Calculate tumor volumes across slices."""
    return torch.sum(labels, axis=(0, 1, 2)).float()

def get_slice_indices(z_mask_size, top_k=3):
    """Get indices of top K slices by tumor volume."""
    _, indices = torch.topk(z_mask_size, top_k)
    return indices.squeeze()
