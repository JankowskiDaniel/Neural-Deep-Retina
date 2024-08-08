import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.wrappers import MetricTracker
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Tuple

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
    save_outputs_and_targets: bool = True,
    save_dir: Path = Path("predicitons"),
) -> Tuple[float, dict]:
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
        metrics_dict[f"Loss: {loss_fn.__class__.__name__}"] = test_loss

        if save_outputs_and_targets:
            outputs_df.to_csv(save_dir / "outputs.csv", index=False)
            targets_df.to_csv(save_dir / "targets.csv", index=False)

    return test_loss, metrics_dict
