import numpy as np


class EarlyStopping:
    """
    Class for implementing early stopping during training.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 3.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.0.
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Counter to keep track of the number of epochs with no improvement.
        min_loss (float): Minimum loss value observed so far.
    Methods:
        __call__(self, loss: float) -> bool:
            Checks if the given loss value qualifies for early stopping.
    """  # noqa: E501

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def __call__(self, loss: float) -> bool:
        """
        Checks if the given loss value qualifies for early stopping.
        Args:
            loss (float): The loss value to be checked.
        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        if loss + self.min_delta < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss + self.min_delta > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
