import matplotlib.pyplot as plt
from pathlib import Path


def visualize_loss(
    train_history: dict,
    results_dir: Path,
    filename: str = "loss.png",
):
    """
    Visualizes the training and validation loss.

    Args:
        train_history (dict): The training history containing the training and validation loss.

    Returns:
        None
    """  # noqa: E501
    save_path = results_dir / "plots" / filename

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_history["train_loss"], label="Train loss")
    ax.plot(train_history["valid_loss"], label="Validation loss")
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
