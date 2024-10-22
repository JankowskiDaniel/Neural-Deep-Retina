from pathlib import Path
from typing import Any
from interfaces.base_handler import BaseHandler
from data_handlers import (
    H5Dataset,
    H5SeqDataset,
    BaselineRGBDataset,
    BaselineSeqRGBDataset,
)
from data_models.config_models import DataConfig

HANDLERS: dict[str, BaseHandler] = {
    "H5Dataset": H5Dataset,
    "H5SeqDataset": H5SeqDataset,
    "BaselineRGBDataset": BaselineRGBDataset,
    "BaselineSeqRGBDataset": BaselineSeqRGBDataset,
}


def load_data_handler(
    data_config: DataConfig,
    results_dir: Path,
    is_train: bool = True,
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
    # Initialize the handler class with remaining parameters
    parsed_config = data_config.dict(
        exclude={"data_handler", "img_shape", "prediction_step", "subset_size"}
    )
    dataset = handler_class(
        results_dir=results_dir,
        is_train=is_train,
        y_scaler=y_scaler,
        use_saved_scaler=use_saved_scaler,
        prediction_step=prediction_step,
        subset_size=subset_size,
        **parsed_config,
    )
    return dataset
