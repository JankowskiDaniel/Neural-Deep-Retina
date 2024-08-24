import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.wrappers import MetricTracker
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Tuple

from models import DeepRetinaModel
from visualize.visualize_dataset import visualize_output_images


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
    for images in train_loader:
        model.zero_grad()
        images = images.to(device)
        outputs = model(images)

        # Compare input images with output images
        loss = loss_fn(outputs, images)

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
    epoch: int,
    results_dir: Path,
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
        for i, images in enumerate(valid_loader):
            images = images.to(device)
            outputs = model(images)

            # Compare input images with output images
            loss = loss_fn(outputs, images)
            valid_batch_losses.append(loss.item())

            # Plot the first batch of images
            if epoch % 5 == 0 and i == 0:
                n = 10
                visualize_output_images(
                    (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)[
                        :n
                    ],
                    (outputs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)[
                        :n
                    ],
                    epoch,
                    i + 1,
                    results_dir,
                    n=n,
                    batch_type="valid",
                )
        valid_loss = np.sum(valid_batch_losses) / len(valid_batch_losses)
    return valid_loss


def test_model(
    model: DeepRetinaModel,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: Literal["cuda", "cpu"],
    tracker: MetricTracker,
) -> Tuple[float, dict]:
    """
    Test the given model on the test data.
    Args:
        model (DeepRetinaModel): The model to be tested.
        test_loader (DataLoader): The data loader for the test data.
        loss_fn (nn.Module): The loss function to calculate the test loss.
        device (Literal["cuda", "cpu"]): The device to run the test on.
        tracker (MetricTracker): The metric tracker to track the evaluation metrics.
    Returns:
        Tuple[float, dict]: A tuple containing the test loss and a dictionary of evaluation metrics.
    """  # noqa: E501
    model.eval()
    test_losses = []
    with torch.no_grad():
        tracker.increment()
        for images in (pbar := tqdm(test_loader, unit="batch")):
            images = images.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, images)
            pbar.set_description(f"Test loss: {str(loss.item())}")
            test_losses.append(loss.item())
            tracker.update(outputs, images)

        metrics_dict = tracker.cpu().compute_all()
        test_loss = np.sum(test_losses) / len(test_losses)
        metrics_dict[f"Loss: {loss_fn.__class__.__name__}"] = test_loss

    return test_loss, metrics_dict
