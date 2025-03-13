import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.wrappers import MetricTracker
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Tuple, Any

from models import DeepRetinaModel
from utils.metrics import (
    compute_pearson_correlations,
    compute_wasserstein_distances,
)


def train_epoch(
    model: DeepRetinaModel,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: Literal["cuda", "cpu"],
    epoch: int,
) -> float:
    """
    Trains the model for one epoch using the given data loader and optimizer.
    Args:
        model (DeepRetinaModel): The neural network model to train.
        train_loader (DataLoader): The data loader containing the training data.
        optimizer (Optimizer): The optimizer used to update the model's parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (Literal["cuda", "cpu"]): The device to use for training (e.g., "cuda" for GPU or "cpu" for CPU).
        epoch (int): The current epoch number.
    Returns:
        float: The average training loss for the epoch.
    """  # noqa: E501
    model.train()
    train_batch_losses = []
    for data, labels in train_loader:
        model.zero_grad()
        images = data.to(device)
        targets = labels.to(device)
        outputs = model(images)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_batch_losses.append(loss.item())
    train_loss = np.sum(train_batch_losses) / len(train_batch_losses)
    return train_loss


def valid_epoch(
    model: DeepRetinaModel,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: Literal["cuda", "cpu"],
) -> float:
    """
    Calculates the average validation loss for a given model.
    Args:
        model (DeepRetinaModel): The model to be evaluated.
        valid_loader (DataLoader): The data loader for the validation dataset.
        loss_fn (nn.Module): The loss function used to calculate the loss.
        device (Literal["cuda", "cpu"]): The device on which the model and data are located.
    Returns:
        float: The average validation loss.
    """  # noqa: E501
    model.eval()
    valid_batch_losses = []
    with torch.no_grad():
        for data, labels in valid_loader:
            images = data.to(device)
            targets = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, targets)
            valid_batch_losses.append(loss.item())
        valid_loss = np.sum(valid_batch_losses) / len(valid_batch_losses)
    return valid_loss


def test_model(
    model: DeepRetinaModel,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: Literal["cuda", "cpu"],
    tracker: MetricTracker,
    save_outputs_and_targets: bool = True,
    save_dir: Path = Path("predicitons"),
    y_scaler: Any = None,
) -> Tuple[float, dict]:
    """
    Test the given model on the test data.
    Args:
        model (DeepRetinaModel): The model to be tested.
        test_loader (DataLoader): The data loader for the test data.
        loss_fn (nn.Module): The loss function to calculate the test loss.
        device (Literal["cuda", "cpu"]): The device to run the test on.
        tracker (MetricTracker): The metric tracker to track the evaluation metrics.
        save_outputs_and_targets (bool, optional): Whether to save the outputs and targets. Defaults to True.
        save_dir (Path, optional): The directory to save the outputs and targets. Defaults to Path("predicitons").
    Returns:
        Tuple[float, dict]: A tuple containing the test loss and a dictionary of evaluation metrics.
    """  # noqa: E501
    model.eval()
    test_losses = []
    outputs_df = pd.DataFrame()  # Create an empty dataframe for outputs
    targets_df = pd.DataFrame()  # Create an empty dataframe for targets
    with torch.no_grad():
        tracker.increment()
        for data, labels in (pbar := tqdm(test_loader, unit="batch")):
            images = data.to(device)
            targets = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, targets)
            pbar.set_description(f"Test loss: {str(loss.item())}")
            test_losses.append(loss.item())
            tracker.update(outputs, targets)

            if save_outputs_and_targets:
                outputs_df = pd.concat(
                    [outputs_df, pd.DataFrame(outputs.cpu().numpy())]
                )
                targets_df = pd.concat(
                    [targets_df, pd.DataFrame(targets.cpu().numpy())]
                )

        metrics_dict = tracker.cpu().compute_all()
        test_loss = np.sum(test_losses) / len(test_losses)
        metrics_dict["test_loss"] = test_loss
        # Report RMSE if MSE is present
        if "MeanSquaredError" in metrics_dict:
            metrics_dict["RootMeanSquaredError"] = np.sqrt(
                metrics_dict["MeanSquaredError"]
            )

        if save_outputs_and_targets:
            # Save scaled outputs and targets
            outputs_df.to_csv(save_dir / "scaled_outputs.csv", index=False)
            targets_df.to_csv(save_dir / "scaled_targets.csv", index=False)
            # Rescale the outputs and targets if a scaler is provided
            corr_data_mode = "scaled"
            if y_scaler is not None:
                corr_data_mode = "unscaled"
                outputs_df = pd.DataFrame(
                    y_scaler.inverse_transform(outputs_df)
                )
                targets_df = pd.DataFrame(
                    y_scaler.inverse_transform(targets_df)
                )
                # Save unscaled outputs and targets
                outputs_df.to_csv(
                    save_dir / "unscaled_outputs.csv", index=False
                )
                targets_df.to_csv(
                    save_dir / "unscaled_targets.csv", index=False
                )
                # Calculate MSE on the unscaled data
                mse = np.mean((outputs_df.values - targets_df.values) ** 2)
                metrics_dict["MSE_unscaled"] = mse
                metrics_dict["RMSE_unscaled"] = np.sqrt(mse)
                # Calculate MAE on the unscaled data
                mae = np.mean(np.abs(outputs_df.values - targets_df.values))
                metrics_dict["MAE_unscaled"] = mae
            # Calculate Pearson correlation between outputs and targets
            pearson_correlations = compute_pearson_correlations(
                outputs_df, targets_df, prefix="pcorr_" + corr_data_mode + "_"
            )
            # Add Pearson correlations to metrics_dict
            metrics_dict.update(pearson_correlations)
            # Calculate Wasserstein distance between outputs and targets
            wasserstein_distances = compute_wasserstein_distances(
                outputs_df.values, targets_df.values, "emd_" + corr_data_mode + "_"
            )
            # Add Wasserstein distances to metrics_dict
            metrics_dict.update(wasserstein_distances)

    return test_loss, metrics_dict
