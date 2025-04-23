from abc import ABC, abstractmethod
import torch.nn as nn


class Predictor(nn.Module, ABC):
    """
    Abstract class for predictors. All predictors should inherit from this class.
    """

    def __init__(self, **kwargs) -> None:
        super(Predictor, self).__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
