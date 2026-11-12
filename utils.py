import torch
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


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

def dice_np(pred, mask, eps=1e-6):
    # pred, mask: (H, W) bool arrays
    inter = (pred & mask).sum()
    return (2 * inter + eps) / (pred.sum() + mask.sum() + eps)

def postprocess(pred):
    # pred: (H, W) numpy array with 0/1
    pred = pred.astype(np.uint8)

    # 1) keep only the largest connected region (removes islands)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    if n > 1:  # label 0 = background
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred = (labels == largest).astype(np.uint8)

    # 2) fill holes inside the femur
    pred = binary_fill_holes(pred).astype(np.uint8)

    return pred