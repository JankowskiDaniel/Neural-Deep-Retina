import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd
from yaml import safe_load

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
        n_channels: int = dataset.output_shape[0]
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
            Input shape {dataset.input_shape} Output shape {dataset.output_shape}\n"
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)


def visualize_outputs_and_targets(
    targets: pd.DataFrame,
    outputs: pd.DataFrame,
    plots_dir: Path,
    file_name: str = "outputs_and_targets.png",
    is_train: bool = False,
    return_fig: bool = False,
) -> plt.Figure | None:

    n_channels: int = targets.shape[-1]
    fig, _ = plt.subplots(
        ncols=1,
        nrows=n_channels,
        figsize=(10, 2 * n_channels),
        sharex=True,
        squeeze=False,
    )
    for channel, ax in enumerate(fig.axes):
        sns.lineplot(
            outputs.iloc[:, channel],
            ax=ax,
            linewidth=0.5,
            label="model output",
        )
        sns.lineplot(targets.iloc[:, channel], ax=ax, linewidth=0.5, label="target")
        ax.set_title(f"Channel {channel}")
        # Show only one legend
        if channel == 0:
            lines, labels = ax.get_legend_handles_labels()
            fig.legend(lines, labels, loc="upper right")
        ax.get_legend().remove()
    fig.supylabel("Output signal")
    fig.supxlabel("Time")
    prefix = "Train" if is_train else "Test"
    fig.suptitle(f"{prefix} data model output and target\n")

    fig.tight_layout()
    fig.savefig(plots_dir / file_name, dpi=150)
    if return_fig:
        return fig
    return None


if __name__ == "__main__":

    # To tun this script, enter src in the terminal
    # and run python -m visualize.visualize_dataset

    path = Path("../results/my_training/predictions")

    # visualize_outputs_and_targets(path, path)

    y_scaler = MinMaxScaler()
    with open("../config.yaml", "r") as stream:
        config = safe_load(stream)

    path = ".." / Path(config["DATA"]["path"])

    # load the dataset
    train_dataset = H5Dataset(
        path=path,
        response_type="firing_rate_10ms",
        is_train=True,
        is_rgb=False,
        y_scaler=y_scaler,
        results_dir=Path("visualize"),
    )

    visualize_target(train_dataset)
