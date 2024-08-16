from dataclasses import dataclass


@dataclass
class DataConfig:
    img_shape: list[int]
    rgb: bool
    path: str


@dataclass
class EncoderConfig:
    name: str
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


@dataclass
class Config:
    data: DataConfig
    training: TrainingConfig
    testing: TestingConfig
