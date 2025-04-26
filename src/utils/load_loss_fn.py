import torch.nn as nn
import torch
import numpy as np
from utils.loss_functions import (
    BinaryPDFWeightedBCEWithLogitsLoss,
    FrequencyWeightedMSELoss,
    SingnalWeightedMSELoss,
)


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "bce": nn.BCEWithLogitsLoss,
    "bce_weighted": nn.BCEWithLogitsLoss,
    "bce_pdf_weighted": BinaryPDFWeightedBCEWithLogitsLoss,
    "swmse": SingnalWeightedMSELoss,
    "fweighted_mse": FrequencyWeightedMSELoss,
}


def load_loss_function(
    loss_fn_name: str, target: np.ndarray, device: str
) -> nn.Module:
    """
    Load a loss function by name.

    Args:
        loss_fn_name (str): The name of the loss function to load.
        target (np.ndarray): The target data used for loss function initialization.
        device (str): The device to which the loss function should be moved.

    Returns:
        nn.Module: The loaded loss function.
    """

    if loss_fn_name == "bce_weighted":
        pos_weights = compute_pos_weights(target, is_discretized=True)
        pos_weights = pos_weights.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif loss_fn_name == "bce_pdf_weighted":
        pos_weights = compute_pos_weights(target, is_discretized=False)
        pos_weights = pos_weights.to(device)
        loss_fn = BinaryPDFWeightedBCEWithLogitsLoss(pos_weights)
    elif loss_fn_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{loss_fn_name}' is not supported.")
    else:
        loss_fn = LOSS_FUNCTIONS[loss_fn_name]()
    loss_fn = loss_fn.to(device)
    return loss_fn


def compute_pos_weights(
    target: np.ndarray, is_discretized: bool = True, threshold: float = 0.1
) -> torch.Tensor:
    """
    Compute positive weights for the target data.

    Args:
        target (np.ndarray): The target data.
        is_discretized (bool): Flag indicating whether the target is discretized.
        threshold (float): The threshold for positive samples.

    Returns:
        torch.Tensor: The computed positive weights.
    """

    if not is_discretized:
        # If the target is not discretized, apply binarization
        target = np.where(target > threshold, 1, 0)

    # Count of positive (1s) per class
    pos_counts = np.sum(target, axis=1)
    # Count of negative (0s) per class
    neg_counts = target.shape[1] - pos_counts
    # Compute pos_weight (negatives / positives), ensuring no division by zero
    pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    # Transform to tensor
    pos_weights = torch.tensor(pos_weight, dtype=torch.float32)

    return pos_weights
