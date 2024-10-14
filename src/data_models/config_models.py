from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel


class DataConfig(BaseModel, extra="allow"):  # type: ignore
    data_handler: str
    img_shape: list[int]
    path: str
    response_type: Literal["firing_rate_10ms", "binned"]
    prediction_step: int
    subset_size: int


@dataclass
class EncoderConfig:
    name: str
    weights: str
    freeze: bool
    learning_rate: float


@dataclass
class PredictorConfig:
    name: str
    learning_rate: float


@dataclass
class TrainingConfig:
    encoder: EncoderConfig
    predictor: PredictorConfig
    batch_size: int
    epochs: int
    num_units: int
    early_stopping: bool
    early_stopping_patience: int
    save_logs: bool


@dataclass
class TestingConfig:
    batch_size: int
    weights: str
    metrics: list[str]
    save_logs: bool
    run_on_train_data: bool


@dataclass
class Config:
    data: DataConfig
    training: TrainingConfig
    testing: TestingConfig
