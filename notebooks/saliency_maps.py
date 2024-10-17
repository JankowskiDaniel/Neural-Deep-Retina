import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append("src")

from utils.logger import get_logger
from utils import (
    load_config,
    load_model,
    get_metric_tracker,
    load_data_handler,
)


results_dir = Path("og_encoder_exp_1")

# Create path object to results directory
results_dir_path = "results" / results_dir
# Load config from the results directory
config_path = results_dir_path / "config.yaml"

config = load_config(config_path)

logger = get_logger(
    log_to_file=config.testing.save_logs,
    log_file=results_dir_path / "test_logs.log",
)

logger.info("Preparing to test model...")
logger.info(f"Using config file: {config_path}")
weights_path = results_dir_path / "models" / config.testing.weights
logger.info(f"Using model: {weights_path}")

# load the model
logger.info("Loading model...")
model = load_model(config)
model.load_state_dict(torch.load(weights_path))

test_dataset = load_data_handler(
    config.data,
    results_dir=results_dir_path,
    is_train=False,
    y_scaler=StandardScaler(),
    use_saved_scaler=True,
)

# Get sample data to check dimensions
X, y = test_dataset[0]
logger.info(f"Input data point shape: {X.shape}")
logger.info(f"Target data point shape: {y.shape}")

TORCH_SEED = 12
logger.info(f"Manually set PyTorch seed: {TORCH_SEED}")
torch.manual_seed(TORCH_SEED)

logger.info(f"Test dataset size: {len(test_dataset)}")

# Define training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
PIN_MEMORY = False  # torch.cuda.is_available()
NUM_WORKERS = 0  # if torch.cuda.is_available() else 0
BATCH_SIZE = config.testing.batch_size

# Define data loaders
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
)

# Define loss function
loss_fn = nn.MSELoss()

# Create metric tracker
metrics_tracker = get_metric_tracker(config.testing.metrics)
metrics_tracker.to(DEVICE)

# Set the path for saving predictions
predictions_dir = results_dir_path / "testset_predictions"
# Set the path for saving plots
plots_dir = results_dir_path / "plots"

model = model
model.eval()


def get_channel_importance(X: torch.tensor, n_units: int) -> torch.tensor:
    # we need to find the gradient with respect to the input image
    # so we need to call requires_grad_ on it
    X = X.unsqueeze(0)
    X.requires_grad_()
    scores = model(X)

    channel_importances = []
    for c in range(n_units):
        # Forward pass
        # Backward pass for channel c
        scores[0][c].backward(retain_graph=True)
        # Sum absolute gradient values for each channel
        channel_importance = torch.sum(X.grad.data.abs(), dim=(2, 3))
        # Normalize channel importance
        channel_importance = channel_importance / torch.sum(channel_importance, dim=1)
        channel_importances.append(channel_importance)

    return torch.stack(channel_importances)


# How many images to use for importance calculation
N_SAMPLES = 1000
# Generate random image indices
indices = np.random.choice(len(test_dataset), N_SAMPLES, replace=False)
n_units = config.training.num_units
importances = torch.Tensor()
for idx in indices:
    X, y = test_dataset[idx]
    ci = get_channel_importance(X, n_units=n_units)
    importances = torch.cat((importances, ci), dim=1)

# Calculate mean importance for channels
mean_importance = importances.mean(dim=1)
std_importance = importances.std(dim=1)

print(mean_importance.shape, std_importance.shape)

x_labels = range(1, mean_importance.shape[1] + 1)

fig, _ = plt.subplots(
    ncols=1,
    nrows=n_units,
    figsize=(10, 2 * n_units),
    squeeze=False,
)
for channel, ax in enumerate(fig.axes):
    ax.bar(x_labels, mean_importance[channel], yerr=std_importance[channel])
    ax.set_title(f"Channel {channel}")
fig.supylabel("Importance")
fig.supxlabel(f"Number of frame in sequence\nThe most recent image is {x_labels[-1]}")
fig.suptitle("Gradient-based mean channel importance (with std)")

fig.tight_layout()
fig.savefig("notebooks\importances.png", dpi=150)
