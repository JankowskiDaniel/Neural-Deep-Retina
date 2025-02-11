from torch.nn import Module


def count_parameters(model: Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (Module): The PyTorch model for which to count the parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
