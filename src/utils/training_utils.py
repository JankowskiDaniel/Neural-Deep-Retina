from typing import Literal, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchmetrics.wrappers import MetricTracker

from models import DeepRetinaModel


def train_epoch(
    model: DeepRetinaModel,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: Literal["cuda", "cpu"],
    epoch: int,
) -> float:
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
):
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
) -> Tuple[float, dict]:
    model.eval()
    test_losses = []
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
        metrics_dict = tracker.cpu().compute_all()
        test_loss = np.sum(test_losses) / len(test_losses)
        metrics_dict[f"Loss: {loss_fn.__class__.__name__}"] = test_loss
    return test_loss, metrics_dict
