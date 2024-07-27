from dataclasses import dataclass

@dataclass
class DataConfig:
    img_size: list[int]
    rgb: bool
    path: str


@dataclass
class NNConfig:
    name: str
    learning_rate: float


@dataclass
class TrainingConfig:
    encoder: NNConfig
    model: NNConfig
    batch_size: int
    epochs: int
    num_units: int


@dataclass
class Config:
    data: DataConfig
    training: TrainingConfig