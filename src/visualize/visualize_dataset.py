import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd

from data_handlers import H5Dataset


def visualize_target(
    dataset: H5Dataset, save_path: Path = Path("dataset_target.png")
) -> None:
    """
    Visualizes the target of a dataset.

    Args:
        dataset (H5Dataset): The dataset containing the target to visualize.
        save_path (Path, optional): The path to save. Defaults to "dataset_target.png".
    """
    if dataset.Y is not None:
        n_channels: int = dataset.target_shape[0]
        fig, axes = plt.subplots(
            ncols=1,
            nrows=n_channels,
            figsize=(10, 2 * n_channels),
            sharex=True,
            squeeze=False,
        )
        for channel, ax in enumerate(axes.reshape(-1)):
            sns.lineplot(dataset.Y[channel], ax=ax, linewidth=0.5)
            ax.set_title(f"Channel {channel}")
        fig.supylabel("Output signal")
        fig.supxlabel("Time")
        fig.suptitle(
            f"dataset targets plot\n{dataset.file_path}\n \
            Input shape {dataset.input_shape} Output shape {dataset.target_shape}\n"
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)


def visualize_outputs_and_targets(
    predictions_dir: Path,
) -> None:
    """
    Visualizes the model outputs and targets by plotting them on separate channels.

    Args:
        predictions_dir (Path): The directory containing the predictions.

    Returns:
        None
    """

    targets = pd.read_csv(predictions_dir / "targets.csv", header=0)
    outputs = pd.read_csv(predictions_dir / "outputs.csv", header=0)
    n_channels: int = targets.shape[-1]
    fig, axes = plt.subplots(
        ncols=1,
        nrows=n_channels,
        figsize=(10, 2 * n_channels),
        sharex=True,
        squeeze=False,
    )
    for channel, ax in enumerate(axes.reshape(-1)):
        sns.lineplot(outputs.iloc[:, channel], ax=ax, linewidth=0.5, label="Output")
        sns.lineplot(targets.iloc[:, channel], ax=ax, linewidth=0.5, label="Target")
        ax.set_title(f"Channel {channel}")
    fig.supylabel("Output signal")
    fig.supxlabel("Time")
    fig.suptitle("Model output and target plot\n")

    # Show only one legend
    handles, labels = axes.reshape(-1)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")

    fig.tight_layout()
    fig.savefig(predictions_dir / "outputs_and_targets.png", dpi=150)


if __name__ == "__main__":

    y_scaler = MinMaxScaler()
    path = Path("../data/neural_code_data/ganglion_cell_data/15-10-07/naturalscene.h5")

    # load the dataset
    train_dataset = H5Dataset(
        path=path,
        response_type="firing_rate_10ms",
        is_train=True,
        is_rgb=False,
        y_scaler=y_scaler,
    )

    visualize_target(train_dataset)
