import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from pathlib import Path
from utils.loss_functions import (
    BinaryPDFWeightedBCEWithLogitsLoss,
    FrequencyWeightedMSELoss,
    SignalWeightedMSELoss,
    TverskyLossMultiLabel,
)


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "bce": nn.BCEWithLogitsLoss,
    "bce_weighted": nn.BCEWithLogitsLoss,
    "bce_pdf_weighted": BinaryPDFWeightedBCEWithLogitsLoss,
    "swmse": SignalWeightedMSELoss,
    "fweighted_mse": FrequencyWeightedMSELoss,
    "tversky": TverskyLossMultiLabel,
}


def load_loss_function(
    loss_fn_name: str,
    device: str,
    results_dir: Path,
    target: Optional[np.ndarray],
    is_train: bool = True,
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

    if loss_fn_name == "bce_weighted" or loss_fn_name == "bce_pdf_weighted":
        loss_fn = load_weighted_bce_loss(
            loss_fn_name=loss_fn_name,
            device=device,
            results_dir=results_dir,
            target=target,
            is_train=is_train,
        )
    elif loss_fn_name == "tversky":
        loss_fn = TverskyLossMultiLabel(alpha=0.3, beta=0.7, eps=1e-8)
    elif loss_fn_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{loss_fn_name}' is not supported.")
    else:
        loss_fn = LOSS_FUNCTIONS[loss_fn_name]()
    loss_fn = loss_fn.to(device)
    return loss_fn


def load_weighted_bce_loss(
    loss_fn_name: str,
    device: str,
    results_dir: Path,
    target: Optional[np.ndarray],
    is_train: bool = True,
) -> nn.Module:

    if is_train:
        if target is None:
            raise ValueError(
                "Target data must be provided for pos_weights computation."
            )
        pos_weights = compute_pos_weights(
            target,
            is_discretized=True if loss_fn_name == "bce_weighted" else False,
        )
        save_pos_weights(pos_weights, results_dir)
    else:
        pos_weights = load_pos_weights(results_dir)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    pos_weights_tensor = pos_weights_tensor.to(device)
    if loss_fn_name == "bce_weighted":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    else:
        loss_fn = BinaryPDFWeightedBCEWithLogitsLoss(pos_weights_tensor)
    return loss_fn


def compute_pos_weights(
    target: np.ndarray, is_discretized: bool = True, threshold: float = 0.1
) -> np.ndarray:
    """
    Compute positive weights for the target data.

    Args:
        target (np.ndarray): The target data.
        is_discretized (bool): Flag indicating whether the target is discretized.
        threshold (float): The threshold for positive samples.

    Returns:
        np.ndarray: The computed positive weights.
    """

    if not is_discretized:
        # If the target is not discretized, apply binarization
        target = np.where(target > threshold, 1, 0)

    # Count of positive (1s) per class
    pos_counts = np.sum(target, axis=1)
    # Count of negative (0s) per class
    neg_counts = target.shape[1] - pos_counts
    # Compute pos_weight (negatives / positives), ensuring no division by zero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)

    return pos_weights


def save_pos_weights(
    pos_weights: np.ndarray,
    results_dir: Path,
) -> None:
    """
    Save the positive weights to a file.

    Args:
        pos_weights (np.ndarray): The positive weights to save.
        results_dir (Path): The directory where the file will be saved.
    """
    np.savetxt(results_dir / "pos_weights.txt", pos_weights)


def load_pos_weights(
    results_dir: Path,
) -> np.ndarray:
    """
    Load the positive weights from a file.

    Args:
        results_dir (Path): The directory where the file is saved.

    Returns:
        np.ndarray: The loaded positive weights.
    """
    pos_weights = np.loadtxt(results_dir / "pos_weights.txt")
    return pos_weights
