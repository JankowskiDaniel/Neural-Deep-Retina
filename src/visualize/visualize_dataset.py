import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from data_handlers import H5Dataset


def visualize_outputs(
    dataset: H5Dataset, save_path: Path = Path("dataset_outputs.png")
) -> None:
    """
    Visualizes the outputs of a dataset.

    Args:
        dataset (H5Dataset): The dataset containing the outputs to visualize.
        save_path (Path, optional): The path to save. Defaults to "dataset_outputs.png".
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
            f"dataset output plot\n{dataset.file_path}\n \
            Input shape {dataset.input_shape} Output shape {dataset.output_shape}\n"
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)


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

    visualize_outputs(train_dataset)
