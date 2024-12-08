import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2).item()
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(pred, target):
    """Structural Similarity Index (SSIM)."""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=target_np.max() - target_np.min())
