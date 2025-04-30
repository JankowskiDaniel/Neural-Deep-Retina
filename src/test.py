import torch
from torch.utils.data import DataLoader
from time import time
import pandas as pd
from pathlib import Path
from data_models.config_models import Config
from utils.training_utils import test_model
from utils.logger import get_logger
from utils import (
    load_model,
    get_metric_tracker,
    load_data_handler,
    load_loss_function,
)
from utils.classification_metrics import (
    create_classification_report,
    save_classification_report,
)
from visualize.visualize_dataset import visualize_outputs_and_targets
import wandb
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def test(config: Config) -> None:

    # Assumes the config path was overriden in the command line
    results_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()
        .runtime.config_sources[1]
        .path
    )

    logger = get_logger(
        log_to_file=config.testing.save_logs,
        log_file=results_dir / "test_logs.log",
    )

    logger.info(f"Results dir:{str(results_dir)}")
    logger.info("Preparing to test model...")
    weights_path = results_dir / "models" / config.testing.weights
    logger.info(f"Using model: {weights_path}")

    # load the model
    logger.info("Loading model...")
    model = load_model(config)
    model.load_state_dict(torch.load(weights_path))

    # For calculating pos_weights if necessary
    train_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
        is_train=True,
        use_saved_scaler=True,
    )

    test_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
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
    with open(results_dir / "id.txt", "r") as f:
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
    loss_fn_name = config.training.loss_function
    loss_fn = load_loss_function(
        loss_fn_name=loss_fn_name,
        target=train_dataset.get_target(),
        device=DEVICE,
    )

    # Create metric tracker
    metrics_tracker = get_metric_tracker(config.testing.metrics, DEVICE=DEVICE)

    # Set the path for saving predictions
    predictions_dir = results_dir / "testset_predictions"
    # Set the path for saving plots
    plots_dir = results_dir / "plots"

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
        is_classification=config.data.is_classification,
    )
    total_time = time() - start_testing_time
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Total testing time: {total_time:.2f} seconds")

    outputs = pd.read_csv(predictions_dir / "unscaled_outputs.csv")
    targets = pd.read_csv(predictions_dir / "unscaled_targets.csv")

    # Log raw values as a W&B Table
    for channel in range(outputs.shape[1]):
        data = list(zip(outputs.iloc[:, channel], targets.iloc[:, channel]))
        table = wandb.Table(data=data, columns=["model_output", "target"])
        wandb.log({f"test_predictions/channel_{channel}": table})

    wandb.log({"TEST_DATA_METRICS": metrics_dict})

    # Create a DataFrame from the metrics dictionary
    df_results = pd.DataFrame(metrics_dict)
    # Save results to a csv file
    df_results.to_csv(results_dir / "test_results.csv", index=False)
    logger.info(f"Results saved to {results_dir / 'test_results.csv'}")

    # Plot outputs and targets
    fig = visualize_outputs_and_targets(
        targets=targets,
        outputs=outputs,
        plots_dir=plots_dir,
        file_name="unscaled_test_outputs_and_targets.png",
        is_train=False,
        return_fig=True,
    )

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

    wandb.log({"Plots/Test_Scaled": fig})
    logger.info(
        f"Outputs and targets visualizations saved to {predictions_dir}"
    )

    if config.data.is_classification:
        clf_report = create_classification_report(
            targets=targets,
            outputs=outputs,
        )
        save_classification_report(
            clf_report=clf_report,
            save_dir=plots_dir,
            file_name="classification_report",
            is_train=False,
        )
        wandb.log({"TEST_CLF_REPORT": clf_report})

    if config.testing.run_on_train_data:
        logger.info("Testing on the training data...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
        )

        # Set the path for saving predictions
        predictions_dir = results_dir / "trainset_predictions"
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
            is_classification=config.data.is_classification,
        )
        total_time = time() - start_testing_time
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Total testing time: {total_time:.2f} seconds")

        outputs = pd.read_csv(predictions_dir / "unscaled_outputs.csv")
        targets = pd.read_csv(predictions_dir / "unscaled_targets.csv")

        wandb.log({"TRAIN_DATA_METRICS": metrics_dict})

        # Create a DataFrame from the metrics dictionary
        df_results = pd.DataFrame(metrics_dict)
        # Save results to a csv file
        df_results.to_csv(
            results_dir / "test_traindata_results.csv", index=False
        )
        logger.info(
            f"Results saved to {results_dir / 'test_traindata_results.csv'}"
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
            clf_report = create_classification_report(
                targets=targets,
                outputs=outputs,
            )
            save_classification_report(
                clf_report=clf_report,
                save_dir=plots_dir,
                file_name="classification_report",
                is_train=True,
            )

        wandb.finish()

        logger.info(
            f"Outputs and targets visualizations saved to {predictions_dir}"
        )


if __name__ == "__main__":

    test()
