from torchmetrics.wrappers import MetricTracker
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


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
