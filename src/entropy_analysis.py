"""
entropy_analysis.py - Spatial entropy of Grad-CAM heatmaps.

H = -sum_i p_i log p_i, where p is the flattened, L1-normalised heatmap.
Higher entropy means attention is spread over more of the image.
"""
import numpy as np
from scipy.stats import entropy as shannon_entropy


def compute_gradcam_entropy(heatmap: np.ndarray) -> float:
    """heatmap: 2-D array (H x W) of non-negative Grad-CAM activations."""
    flat = np.asarray(heatmap, dtype=np.float64).ravel()
    flat = flat / (flat.sum() + 1e-8)
    return float(shannon_entropy(flat))
