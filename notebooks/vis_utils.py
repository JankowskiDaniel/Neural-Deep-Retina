import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(
    data, targets=None, channel_titles=None, figsize_per_subplot=(5, 3)
):
    """
    Plot multiple time series side by side in a grid, each scaled
    to its own min and max values, with optional target time series.

    Parameters:
    - data: List of lists, where each sublist is a time series for one channel.
    - targets: Optional. List of lists, where each sublist is the target time series for one channel.
    - channel_titles: List of strings, titles for each subplot. If None, default titles are used.
    - figsize_per_subplot: Tuple of (width, height) for each subplot.
    - data_color: Color for the data series.
    - target_color: Color for the target series.
    - alpha: Transparency level for the plot lines.

    Returns:
    - None, displays the matplotlib figure.
    """  # noqa: E501
    N = len(data)  # Number of channels
    # T = len(data[0])  # Assuming all channels have the same number of time points

    # Determine the number of rows and columns for the subplot grid
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))

    # Calculate the total figsize
    fig_width = cols * figsize_per_subplot[0]
    fig_height = rows * figsize_per_subplot[1]
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # If no specific titles are provided, use default titles
    if channel_titles is None:
        channel_titles = [f"Channel {i+1}" for i in range(N)]

    alpha = 0.5
    data_color = "blue"
    target_color = "grey"

    # Plot each channel in its subplot
    for i, ax in enumerate(axes.flatten()):
        if i < N:
            # Plot data series
            ax.plot(data[i], color=data_color, alpha=alpha, label="Data")
            # Plot target series if available
            ax.plot(targets[i], color=target_color, alpha=alpha, label="Target")
            # Set y-axis limits based on the combined min and max of data and target
            combined_min = min(min(data[i]), min(targets[i]))
            combined_max = max(max(data[i]), max(targets[i]))

            ax.set_ylim(combined_min, combined_max)
            ax.set_title(channel_titles[i])
            ax.legend()
        else:
            ax.axis("off")  # Turn off axis for empty subplots

    plt.tight_layout()
    plt.show()
