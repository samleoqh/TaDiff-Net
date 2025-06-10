"""Metrics for evaluating model predictions."""
import torch
import numpy as np
from torchmetrics import MeanAbsoluteError as MAE, PeakSignalNoiseRatio as PSNR
from torchmetrics import Dice
from src.evaluation.ssim import SSIM

class MetricsCalculator:
    def __init__(self, device: str, dice_thresholds: list = [0.25, 0.5, 0.75]):
        self.device = device
        self.dice_thresholds = dice_thresholds
        
        # Initialize metrics
        self.dice_metrics = {
            f'dice_{int(t*100)}': Dice(threshold=t).to(device)
            for t in dice_thresholds
        }
        self.mae = MAE().to(device)
        self.psnr = PSNR().to(device)
        self.ssim = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3).to(device)
    
    def calculate_metrics(self, pred_img, gt_img, pred_mask, gt_mask, pred_mask_rf=None, gt_mask_rf=None):
        """Calculate all metrics for a single prediction"""
        metrics = {}
        print(f'pred_img.shape: {pred_img.shape}, gt_img.shape: {gt_img.shape}')
        print(f'pred_mask.shape: {pred_mask.shape}, gt_mask.shape: {gt_mask.shape}')
        # pred_img = torch.mean(pred_img, dim=0)
        # gt_img = torch.mean(gt_img, dim=0)
        # Image metrics
        metrics['ssim'] = self.ssim(self._clamp_255(pred_img), self._clamp_255(gt_img)).cpu().detach().numpy()
        metrics['mae'] = self.mae(pred_img.clamp(-1, 1), gt_img.clamp(-1, 1)).cpu().detach().numpy()
        metrics['psnr'] = self.psnr(self._clamp_255(pred_img), self._clamp_255(gt_img)).cpu().detach().numpy()
        
        # Mask metrics
        for thresh, metric in self.dice_metrics.items():
            metrics[thresh] = metric(pred_mask, gt_mask).cpu().detach().numpy()
            metrics[f'ravd_{thresh.split("_")[1]}'], vol = self.ravd(pred_mask, gt_mask, threshold=float(thresh.split("_")[1])/100)
            metrics[f'vol_{thresh.split("_")[1]}'] = vol
        
        # Reference mask metrics if provided
        if pred_mask_rf is not None and gt_mask_rf is not None:
            metrics['dice_rf'] = self.dice_metrics['dice_50'](pred_mask_rf, gt_mask_rf).cpu().detach().numpy()
            metrics['ravd_rf'], _ = self.ravd(pred_mask_rf, gt_mask_rf, threshold=0.5)
        
        print(f'metrics: {metrics}')
        
        return metrics
    
    def _clamp_255(self, x):
        """Convert tensor to uint8 range"""
        return ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
    
    @staticmethod
    def ravd(pred, truth, threshold=0.5):
        """Calculate relative absolute volume difference"""
        pred = pred.cpu().detach().numpy()
        truth = truth.cpu().detach().numpy()
        pred = (pred>=threshold).astype(bool)
        truth = truth.astype(bool)
        vol1 = np.count_nonzero(pred)+1
        vol2 = np.count_nonzero(truth)+1
        return (vol1-vol2)/float(vol2), vol1-1

def calculate_tumor_volumes(labels):
    """Calculate tumor volumes across slices."""
    return torch.sum(labels, axis=(0, 1, 2)).float()

def get_slice_indices(z_mask_size, top_k=3):
    """Get indices of top K slices by tumor volume."""
    _, indices = torch.topk(z_mask_size, top_k)
    return indices.squeeze()
