from utils import get_testing_arguments, load_config, load_model
from data_handlers import H5Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from time import time
from pathlib import Path
from utils.training_utils import test_model
from utils.logger import get_logger


if __name__ == "__main__":

    results_dir = get_testing_arguments()

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
    logger.info(f'Using model: {results_dir_path / "models" / config.testing.weights}')

    # load the model
    logger.info("Loading model...")
    model = load_model(config)
    model.load_state_dict(
        torch.load(results_dir_path / "models" / config.testing.weights)
    )

    # load the test dataset
    test_dataset = H5Dataset(
        path=Path(config.data.path),
        response_type="firing_rate_10ms",
        is_train=False,
        is_rgb=config.data.rgb,
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
    BATCH_SIZE = config.testing.batch_size

    # Define data loaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define loss function
    loss_fn = nn.MSELoss()

    # Test the model
    logger.info(f"Testing on {DEVICE} using device: {DEVICE_NAME}")
    model.to(DEVICE)
    start_testing_time = time()

    test_loss = test_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    total_time = time() - start_testing_time
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Total testing time: {total_time:.2f} seconds")
