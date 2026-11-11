import torch
import numpy as np
import cv2

def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

def compute_mean_std(df):
    # mean and std over all pixels of the given xrays (grayscale, scaled to 0-1)
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    n_pixels = 0

    for path in df["xrays"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        n_pixels += img.size

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    return mean, std