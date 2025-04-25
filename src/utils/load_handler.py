from pathlib import Path
from typing import Any, Literal
from interfaces.base_handler import BaseHandler
from data_handlers import (
    H5Dataset,
    H5SeqDataset,
    BaselineRGBDataset,
    BaselineSeqRGBDataset,
    CurriculumBaselineRGBDataset,
)
from data_models.config_models import DataConfig

HANDLERS: dict[str, BaseHandler] = {
    "H5Dataset": H5Dataset,
    "H5SeqDataset": H5SeqDataset,
    "BaselineRGBDataset": BaselineRGBDataset,
    "BaselineSeqRGBDataset": BaselineSeqRGBDataset,
    "CurriculumBaselineRGBDataset": CurriculumBaselineRGBDataset,
}


def load_data_handler(
    data_config: DataConfig,
    results_dir: Path,
    is_train: bool = True,
    subset_type: Literal["train", "val", "test"] = "train",
    y_scaler: Any = None,
    use_saved_scaler: bool = False,
) -> BaseHandler:
    """Initialize the dataset handler based on the YAML config."""
    # Extract the DATA block from the config

    # Dynamically get the handler class from the config
    handler_class_name = data_config.data_handler
    handler_class: BaseHandler = HANDLERS.get(handler_class_name)

    if handler_class is None:
        raise ValueError(f"Unknown data handler: {handler_class_name}")

    prediction_step = data_config.prediction_step
    subset_size = data_config.subset_size
    pred_channels = resolve_pred_channels(
        data_config.pred_channels, data_config.num_units
    )
    # Initialize the handler class with remaining parameters
    keys_to_exclude = [
        "data_handler",
        "img_shape",
        "prediction_step",
        "subset_size",
        "pred_channels",
    ]
    parsed_config = {
        k: v for k, v in data_config.items() if k not in keys_to_exclude
    }

    dataset = handler_class(
        results_dir=results_dir,
        is_train=is_train,
        subset_type=subset_type,
        y_scaler=y_scaler,
        use_saved_scaler=use_saved_scaler,
        prediction_step=prediction_step,
        subset_size=subset_size,
        pred_channels=pred_channels,
        **parsed_config,
    )
    return dataset


def resolve_pred_channels(
    pred_channels: str | list[int], n_units: int
) -> list[int]:
    """
    Resolve the prediction channels from a string or a list of integers.
    If a string "all" is provided, it will return all channels.
    If a list of integers is provided, it will return the list.
    """
    if isinstance(pred_channels, str) and pred_channels.lower() == "all":
        return list(range(n_units))
    elif isinstance(pred_channels, list):
        return pred_channels
    else:
        raise ValueError(
            f"Invalid pred_channels: {pred_channels}."
            + "It should be 'all' or a list of integers."
        )
