import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
import pandas as pd
import wandb.plot
from utils.training_utils import test_model
from utils.logger import get_logger
from utils import (
    get_testing_arguments,
    load_config,
    load_model,
    get_metric_tracker,
    load_data_handler,
)
from visualize.visualize_dataset import visualize_outputs_and_targets
from sklearn.preprocessing import StandardScaler
import wandb


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

    # read _id from .txt file
    with open(results_dir_path / "id.txt", "r") as f:
        uuid = f.readline().strip()
    print(uuid)
    wandb.init(
        entity="jankowskidaniel06-put",
        project="Neural Deep Retina",
        id=uuid,
        resume="allow",
    )

    wandb.log({"test_data_length": len(test_dataset)})

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

    # Test the model
    logger.info(f"Testing on {DEVICE} using device: {DEVICE_NAME}")
    model.to(DEVICE)
    start_testing_time = time()

    test_loss, metrics_dict = test_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=DEVICE,
        tracker=metrics_tracker,
        save_outputs_and_targets=True,
        save_dir=predictions_dir,
    )

    outputs = pd.read_csv(predictions_dir / "outputs.csv")
    targets = pd.read_csv(predictions_dir / "targets.csv")

    # Log raw values as a W&B Table
    for channel in range(outputs.shape[1]):
        data = list(zip(outputs.iloc[:, channel], targets.iloc[:, channel]))
        table = wandb.Table(data=data, columns=["model_output", "target"])
        wandb.log({f"test_predictions/channel_{channel}": table})

    total_time = time() - start_testing_time
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Total testing time: {total_time:.2f} seconds")

    # Create a DataFrame from the metrics dictionary
    df_results = pd.DataFrame(metrics_dict)
    logger.info(
        "Results {}".format(df_results.to_string().replace("\n", "\n\t\t\t\t\t"))
    )

    wandb.log({"MSE_TEST": test_loss})

    # Save results to a csv file
    df_results.to_csv(results_dir_path / "test_results.csv", index=False)
    logger.info(f"Results saved to {results_dir_path / 'test_results.csv'}")

    # Plot outputs and targets
    fig = visualize_outputs_and_targets(
        predictions_dir,
        plots_dir,
        file_name="test_outputs_and_targets.png",
        is_train=False,
        return_fig=True,
    )
    wandb.log({"Plots/Test": fig})

    logger.info(f"Outputs and targets visualization saved to {predictions_dir}")

    if config.testing.run_on_train_data:
        logger.info("Testing on the training data...")

        train_dataset = load_data_handler(
            config.data,
            results_dir=results_dir_path,
            is_train=True,
            y_scaler=StandardScaler(),
            use_saved_scaler=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
        )

        # Set the path for saving predictions
        predictions_dir = results_dir_path / "trainset_predictions"
        # Create metric tracker
        metrics_tracker = get_metric_tracker(config.testing.metrics)
        metrics_tracker.to(DEVICE)

        # Test the model
        logger.info("Testing model on the training data...")
        start_testing_time = time()

        test_loss, metrics_dict = test_model(
            model=model,
            test_loader=train_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            tracker=metrics_tracker,
            save_outputs_and_targets=True,
            save_dir=predictions_dir,
        )

        outputs = pd.read_csv(predictions_dir / "outputs.csv")
        targets = pd.read_csv(predictions_dir / "targets.csv")

        # Log raw values as a W&B Table
        for channel in range(outputs.shape[1]):
            data = list(zip(outputs.iloc[:, channel], targets.iloc[:, channel]))
            table = wandb.Table(data=data, columns=["model_output", "target"])
            wandb.log({f"train_predictions/channel_{channel}": table})

        total_time = time() - start_testing_time
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Total testing time: {total_time:.2f} seconds")

        # Create a DataFrame from the metrics dictionary
        df_results = pd.DataFrame(metrics_dict)
        logger.info(
            "Results {}".format(df_results.to_string().replace("\n", "\n\t\t\t\t\t"))
        )

        wandb.log({"MSE_TRAIN": test_loss})
        # Save results to a csv file
        df_results.to_csv(results_dir_path / "test_traindata_results.csv", index=False)
        logger.info(
            f"Results saved to {results_dir_path / 'test_traindata_results.csv'}"
        )
        # Plot outputs and targets
        fig = visualize_outputs_and_targets(
            predictions_dir,
            plots_dir,
            file_name="train_outputs_and_targets.png",
            is_train=True,
            return_fig=True,
        )
        wandb.log({"Plots/Train": fig})
        logger.info(f"Outputs and targets visualization saved to {predictions_dir}")
        wandb.finish()
