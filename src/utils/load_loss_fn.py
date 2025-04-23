import torch.nn as nn


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "bce": nn.BCEWithLogitsLoss,
    "bce_weighted": nn.BCEWithLogitsLoss,
}


def load_loss_function(loss_fn_name: str, **kwargs) -> nn.Module:
    """
    Load a loss function by name.

    Args:
        loss_fn_name (str): The name of the loss function to load.
        **kwargs: Additional arguments to pass to the loss function.

    Returns:
        nn.Module: The loaded loss function.
    """
    if loss_fn_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{loss_fn_name}' is not supported.")

    return LOSS_FUNCTIONS[loss_fn_name](**kwargs)
