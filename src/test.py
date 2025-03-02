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
from utils.classification_metrics import save_classification_report
from visualize.visualize_dataset import visualize_outputs_and_targets
import wandb


if __name__ == "__main__":

    results_dir, if_wandb = get_testing_arguments()

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
        subset_type="test",
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
    if if_wandb:
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
    loss_fn = nn.BCEWithLogitsLoss()

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

    # Get the y_scaler for the test dataset
    y_scaler = test_dataset.get_y_scaler()

    start_testing_time = time()
    test_loss, metrics_dict = test_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=DEVICE,
        tracker=metrics_tracker,
        save_outputs_and_targets=True,
        save_dir=predictions_dir,
        y_scaler=y_scaler,
    )
    total_time = time() - start_testing_time
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Total testing time: {total_time:.2f} seconds")

    outputs = pd.read_csv(predictions_dir / "unscaled_outputs.csv")
    targets = pd.read_csv(predictions_dir / "unscaled_targets.csv")

    if if_wandb:
        # Log raw values as a W&B Table
        for channel in range(outputs.shape[1]):
            data = list(
                zip(outputs.iloc[:, channel], targets.iloc[:, channel])
            )
            table = wandb.Table(data=data, columns=["model_output", "target"])
            wandb.log({f"test_predictions/channel_{channel}": table})

        wandb.log({"TEST_DATA_METRICS": metrics_dict})

    # Create a DataFrame from the metrics dictionary
    df_results = pd.DataFrame(metrics_dict)
    # Save results to a csv file
    df_results.to_csv(results_dir_path / "test_results.csv", index=False)
    logger.info(f"Results saved to {results_dir_path / 'test_results.csv'}")

    # Plot outputs and targets
    fig = visualize_outputs_and_targets(
        targets=targets,
        outputs=outputs,
        plots_dir=plots_dir,
        file_name="unscaled_test_outputs_and_targets.png",
        is_train=False,
        return_fig=True,
    )

    if if_wandb:
        wandb.log({"Plots/Test_Unscaled": fig})

    # Plot results for scaled outputs and targets
    outputs = pd.read_csv(predictions_dir / "scaled_outputs.csv")
    targets = pd.read_csv(predictions_dir / "scaled_targets.csv")

    fig = visualize_outputs_and_targets(
        targets=targets,
        outputs=outputs,
        plots_dir=plots_dir,
        file_name="scaled_test_outputs_and_targets.png",
        is_train=False,
        return_fig=True,
    )

    if config.data.is_classification:
        save_classification_report(
            targets=targets,
            outputs=outputs,
            plots_dir=plots_dir,
            is_train=False,
            file_name="classification_report",
        )
    if if_wandb:
        wandb.log({"Plots/Test_Scaled": fig})
    logger.info(
        f"Outputs and targets visualizations saved to {predictions_dir}"
    )

    if config.testing.run_on_train_data:
        logger.info("Testing on the training data...")

        train_dataset = load_data_handler(
            config.data,
            results_dir=results_dir_path,
            is_train=True,
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
            y_scaler=y_scaler,
        )
        total_time = time() - start_testing_time
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Total testing time: {total_time:.2f} seconds")

        outputs = pd.read_csv(predictions_dir / "unscaled_outputs.csv")
        targets = pd.read_csv(predictions_dir / "unscaled_targets.csv")

        if if_wandb:
            # Log raw values as a W&B Table
            for channel in range(outputs.shape[1]):
                data = list(
                    zip(outputs.iloc[:, channel], targets.iloc[:, channel])
                )
                table = wandb.Table(
                    data=data, columns=["model_output", "target"]
                )
                wandb.log({f"train_predictions/channel_{channel}": table})

            wandb.log({"TRAIN_DATA_METRICS": metrics_dict})

        # Create a DataFrame from the metrics dictionary
        df_results = pd.DataFrame(metrics_dict)
        # Save results to a csv file
        df_results.to_csv(
            results_dir_path / "test_traindata_results.csv", index=False
        )
        logger.info(
            f"Results saved to {results_dir_path / 'test_traindata_results.csv'}"
        )
        # Plot unscaled outputs and targets
        fig = visualize_outputs_and_targets(
            targets=targets,
            outputs=outputs,
            plots_dir=plots_dir,
            file_name="unscaled_train_outputs_and_targets.png",
            is_train=True,
            return_fig=True,
        )
        if if_wandb:
            wandb.log({"Plots/Train_Unscaled": fig})

        # Plot results for scaled outputs and targets
        outputs = pd.read_csv(predictions_dir / "scaled_outputs.csv")
        targets = pd.read_csv(predictions_dir / "scaled_targets.csv")

        fig = visualize_outputs_and_targets(
            targets=targets,
            outputs=outputs,
            plots_dir=plots_dir,
            file_name="scaled_train_outputs_and_targets.png",
            is_train=False,
            return_fig=True,
        )

        if config.data.is_classification:
            save_classification_report(
                targets=targets,
                outputs=outputs,
                plots_dir=plots_dir,
                is_train=True,
                file_name="classification_report",
            )

        if if_wandb:
            wandb.log({"Plots/Train_Scaled": fig})
            wandb.finish()

        logger.info(
            f"Outputs and targets visualizations saved to {predictions_dir}"
        )
