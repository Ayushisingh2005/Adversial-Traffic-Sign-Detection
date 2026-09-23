"""
Feature Squeezing baseline (Xu, Evans & Qi, NDSS 2018).

Idea: compare the classifier's softmax on the raw image against its softmax on
a "squeezed" version. Clean images barely change; adversarial images -- whose
perturbation is fragile -- shift a lot. Score = L1 distance between the two.

This is the comparison baseline required by the project objectives.
"""
import numpy as np
import torch
import torch.nn.functional as F

import config


def bit_depth_reduce(x, bits=config.SQUEEZE_BIT_DEPTH):
    """Quantise each channel to `bits` bits. x in [0,1]."""
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


def median_filter(x, k=config.SQUEEZE_MEDIAN_KERNEL):
    """Simple k x k median filter via unfold."""
    if k <= 1:
        return x
    pad = k // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)          # B,C,H,W,k,k
    patches = patches.contiguous().view(*patches.shape[:4], -1)
    out = patches.median(dim=-1).values
    return out[:, :, : x.shape[2], : x.shape[3]]


@torch.no_grad()
def squeeze_scores(model, x, device=None):
    """Return per-sample max L1 distance across squeezers. Higher = more suspicious."""
    device = device or config.DEVICE
    x = x.to(device)
    p_orig = F.softmax(model(x), dim=1)

    scores = []
    for sq in (bit_depth_reduce, median_filter):
        p_sq = F.softmax(model(sq(x)), dim=1)
        scores.append((p_orig - p_sq).abs().sum(dim=1))

    return torch.stack(scores).max(dim=0).values.cpu().numpy()


def pick_threshold(clean_scores, target_fpr=config.TARGET_FPR):
    """Choose the threshold giving `target_fpr` false positives on clean data."""
    return float(np.quantile(clean_scores, 1.0 - target_fpr))


def predict(scores, threshold):
    """1 = adversarial, 0 = clean."""
    return (scores > threshold).astype(int)
