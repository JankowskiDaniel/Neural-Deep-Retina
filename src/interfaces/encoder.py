from abc import ABC, abstractmethod
import torch.nn as nn


class Encoder(nn.Module, ABC):
    """
    Abstract class for encoders. All encoders should inherit from this class.
    """

    def __init__(self) -> None:
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_output_shape(self):
        raise NotImplementedError
