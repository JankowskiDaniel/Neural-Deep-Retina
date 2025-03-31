import torch
from torch import nn, mean, ones_like
import numpy as np


class SingnalWeightedMSELoss(nn.Module):
    def __init__(self, epsilon: float = 0.001):
        super(SingnalWeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return mean((input - target) ** 2 * (target + self.epsilon))


class FrequencyWeightedMSELoss(nn.Module):
    def __init__(self, threshold: float = 1.3):
        super(FrequencyWeightedMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        weights = ones_like(target)
        weights[target >= self.threshold] = 5
        return (weights * (input - target) ** 2).sum() / weights.sum()


class BinaryPDFWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, threshold: float = 0.1):
        super(BinaryPDFWeightedBCEWithLogitsLoss, self).__init__()
        self.weights = weights
        self.threshold = threshold

    def forward(self, input, target):

        if not (target.size() == input.size()):
            raise ValueError(
                f"Target size ({target.size()}) must "
                + f"be the same as input size ({input.size()})"
            )

        # Targets > threshold are treated as positive class
        mask = target > self.threshold
        weights = mask * self.weights

        pos_loss = (
            target[mask]
            * torch.log(torch.sigmoid(input[mask]))
            * weights[mask]
        ).sum()

        # Targets <= threshold are treated as negative class
        neg_loss = (
            (1 - target[~mask]) * torch.log(1 - torch.sigmoid(input[~mask]))
        ).sum()

        loss = pos_loss + neg_loss

        return -loss / target.numel()


def compute_pos_weights(
    y: np.ndarray, DEVICE: str, threshold: float
) -> torch.Tensor:
    pos_samples = y > threshold
    neg_samples = y <= threshold
    pos_weights = neg_samples.sum(1) / pos_samples.sum(1)
    print(pos_weights.shape)
    # pos_weights = 1 / (pos_weights + 1e-6)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)
    print(f"POS WEIGHT: {pos_weights}")
    return pos_weights
