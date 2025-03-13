from torchmetrics.wrappers import MetricTracker
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd


# Mapping from metric names to torchmetric objects
metrics_dict = {
    "mae": MeanAbsoluteError(),
    "mse": MeanSquaredError(),
}


def get_metric_tracker(metrics: list[str]) -> MetricTracker:
    """
    Creates a metric tracker based on the given list of metrics.

    Parameters:
        metrics (list[str]): A list of metric names.

    Returns:
        MetricTracker: The created metric tracker.

    Raises:
        KeyError: If a metric name is not supported.

    Example:
        metrics = ['mae', 'mse']
        tracker = get_metric_tracker(metrics)
    """
    metric_objects: list = []
    # Add metric objects to the list
    for m in metrics:
        try:
            metric_objects.append(metrics_dict[m])
        except KeyError:
            print(f"Metric {m} not supported.")
    metric_collection = MetricCollection(metric_objects)
    # Create a metric tracker from the metric collection
    tracker = MetricTracker(metric_collection, maximize=False)
    return tracker


def compute_pearson_correlations(
    outputs: pd.DataFrame, targets: pd.DataFrame, prefix: str = "corr_"
) -> dict:
    """
    Compute the Pearson correlation between the outputs and targets.

    Parameters:
        outputs (pd.DataFrame): The DataFrame containing the model outputs.
        targets (pd.DataFrame): The DataFrame containing the target values.
        suffix (str): The prefix to add to the metric names.

    Returns:
        dict: The computed Pearson correlations for each channel
               and the mean for all channels.

    Example:
        pearson_correlations = compute_pearson_correlation(outputs, targets)
    """
    pearson_correlations = {}
    pearson_corr = outputs.corrwith(targets, method="pearson", axis=0)
    # If all values are the same, the correlation is nan
    # Fill nan values with 0
    pearson_corr = pearson_corr.fillna(0)
    # Add Pearson correlation by each target channel to metrics_dict
    for i, corr in enumerate(pearson_corr):
        pearson_correlations[f"{prefix}_ch_{i}"] = corr
    # Compute the mean Pearson correlation
    pearson_correlations[f"{prefix}mean"] = np.mean(
        list(pearson_correlations.values())
    )
    return pearson_correlations


def compute_wasserstein_distances(
    outputs: np.ndarray, targets: np.ndarray, prefix: str = "emd_"
) -> dict:
    """
    Compute the Wasserstein distance between the outputs and targets.

    Parameters:
        outputs (pd.DataFrame): The DataFrame containing the model outputs.
        targets (pd.DataFrame): The DataFrame containing the target values.
        suffix (str): The prefix to add to the metric names.

    Returns:
        dict: The computed Wasserstein distances for each channel
               and the mean for all channels.

    Example:
        wasserstein_distances = compute_wasserstein_distance(outputs, targets)
    """
    wasserstein_distances = {}
    distances_sum = 0.0
    n_channels = outputs.shape[1]
    for channel in range(n_channels):
        distance = wasserstein_distance(
            outputs[:, channel], targets[:, channel]
        )
        wasserstein_distances[f"{prefix}ch_{channel}"] = distance
        distances_sum += distance
    # Compute the mean Wasserstein distance
    wasserstein_distances[f"{prefix}mean"] = distances_sum / n_channels
    return wasserstein_distances
