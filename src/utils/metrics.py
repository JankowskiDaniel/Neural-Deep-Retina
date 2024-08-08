from torchmetrics.wrappers import MetricTracker
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


metrics_dict = {
    "mae": MeanAbsoluteError(),
    "mse": MeanSquaredError(),
}


def get_metric_tracker(metrics: list[str]):
    metric_collection = MetricCollection([metrics_dict[m] for m in metrics])
    tracker = MetricTracker(metric_collection, maximize=False)
    return tracker
