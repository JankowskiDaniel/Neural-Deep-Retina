import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from time import time
import pandas as pd
from pathlib import Path
from torchmetrics.regression import PearsonCorrCoef

from data_models.config_models import Config
from utils.training_utils import (
    train_epoch,
    valid_epoch,
    test_model,
    check_gradients,
)
from utils.logger import get_logger
from utils import (
    load_model,
    get_metric_tracker,
    load_data_handler,
    load_loss_function,
    EarlyStopping,
)
from data_handlers import (
    CurriculumHandler,
    CurriculumDatasets,
    CurriculumDataloaders,
)
from utils.classification_metrics import (
    create_classification_report,
    save_classification_report,
)
from visualize.visualize_dataset import visualize_outputs_and_targets
from visualize.visualize_loss import visualize_loss
import wandb
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def finetune(config: Config) -> None:

    # Assumes the config path was overriden in the command line
    results_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()
        .runtime.config_sources[1]
        .path
    )

    logger = get_logger(
        log_to_file=config.testing.save_logs,
        log_file=results_dir / "finetuning_logs.log",
    )

    logger.info(f"Results dir:{str(results_dir)}")
    logger.info("CLASSIFICATION: preparing to test model...")
    weights_path = results_dir / "models" / config.testing.weights
    logger.info(f"Using model: {weights_path}")

    # load the model
    logger.info("Loading model...")
    model = load_model(config)
    model.load_state_dict(torch.load(weights_path))

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
        device=DEVICE,
        results_dir=results_dir,
        is_train=False,
        target=None,
    )

    # Create metric tracker
    metrics_tracker = get_metric_tracker(config.testing.metrics, DEVICE=DEVICE)

    # Set the path for saving predictions
    predictions_dir = results_dir / "testset_predictions_clf"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # Set the path for saving plots
    plots_dir = results_dir / "plots_clf"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
        is_classification=True,
    )
    total_time = time() - start_testing_time
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Total testing time: {total_time:.2f} seconds")

    outputs = pd.read_csv(predictions_dir / "unscaled_outputs.csv")
    targets = pd.read_csv(predictions_dir / "unscaled_targets.csv")

    # Create a DataFrame from the metrics dictionary
    df_results = pd.DataFrame(metrics_dict)
    # Save results to a csv file
    df_results.to_csv(results_dir / "test_results_clf.csv", index=False)
    logger.info(f"Results saved to {results_dir / 'test_results_clf.csv'}")

    # Plot outputs and targets
    _ = visualize_outputs_and_targets(
        targets=targets,
        outputs=outputs,
        plots_dir=plots_dir,
        file_name="unscaled_test_outputs_and_targets_clf.png",
        is_train=False,
        return_fig=True,
    )

    # Plot results for scaled outputs and targets
    outputs = pd.read_csv(predictions_dir / "scaled_outputs.csv")
    targets = pd.read_csv(predictions_dir / "scaled_targets.csv")

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

    # FINETUNING ####

    logger.info("Finetuning...")

    train_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
        is_train=True,
        use_saved_scaler=True,
    )

    val_dataset = load_data_handler(
        config.data,
        results_dir=results_dir,
        is_train=False,
        subset_type="val",
        use_saved_scaler=True,
        y_scaler=y_scaler,
    )

    # Set the path for saving predictions
    predictions_dir = results_dir / "ft_trainset_predictions"
    # Create metric tracker
    metrics_tracker = get_metric_tracker(config.testing.metrics)
    metrics_tracker.to(DEVICE)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    N_EPOCHS = config.training.epochs
    ENCODER_LR = config.training.encoder.learning_rate
    PREDICTOR_LR = config.training.predictor.learning_rate
    BATCH_SIZE = config.training.batch_size

    # Define data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
    )

    # Create curriculum handler
    curriculum_handler = CurriculumHandler(
        curriculum_dataloaders=CurriculumDataloaders(train_loader, val_loader),
        curriculum_datasets=(
            CurriculumDatasets(train_dataset, val_dataset)
            if config.training.is_curriculum
            else None
        ),
        is_curriculum=config.training.is_curriculum,
        curriculum_schedule=None,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.predictor.parameters(), "lr": PREDICTOR_LR},
        ]
    )

    loss_fn_name = "mse"
    logger.info(f"Loss function: {loss_fn_name}")

    loss_fn = load_loss_function(
        loss_fn_name=loss_fn_name,
        target=train_dataset.get_target(),
        device=DEVICE,
        results_dir=results_dir,
        is_train=True,
    )

    # Modify the predictor activation function
    model.predictor.activation = "sigmoid"

    wandb.config.update({"ft_loss_fn": loss_fn.__class__.__name__})
    train_history: dict = {"train_loss": [], "valid_loss": []}

    if config.training.early_stopping:
        PATIENCE = config.training.early_stopping_patience
        early_stopping = EarlyStopping(patience=PATIENCE)

    val_metric_tracker = get_metric_tracker(
        ["mae", "mse", "pcorr"],
        initialized_metrics=[
            PearsonCorrCoef(num_outputs=config.data.num_units).to(DEVICE)
        ],
        DEVICE=DEVICE,
    )

    # ########### START TRAINING ###########
    logger.info(f"Training on {DEVICE} using device: {DEVICE_NAME}")
    model.to(DEVICE)
    start_training_time = time()

    best_pcorr = float("-inf")
    for epoch in tqdm(range(N_EPOCHS)):
        start_epoch_time = time()
        train_loader, val_loader = curriculum_handler.get_dataloaders(epoch)
        # training
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
        )
        train_history["train_loss"].append(train_loss)

        wandb.log({"training/ft_train_loss": train_loss, "epoch": epoch})

        if config.training.debug_mode:
            check_gradients(model, logger)

        # validation
        valid_loss, val_metrics = valid_epoch(
            model=model,
            valid_loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            metric_tracker=val_metric_tracker,
        )
        train_history["valid_loss"].append(valid_loss)

        logger.info(
            f"Epoch: {epoch + 1}/{N_EPOCHS} \t Train Loss: {train_loss} | "
            + f"Validation Loss: {valid_loss}"
        )
        logger.info(
            f"Learing rates: Encoder {optimizer.param_groups[0]['lr']} | "
            + f"Predictor {optimizer.param_groups[1]['lr']}"
        )
        # Calculate mean Pearson correlation
        val_mean_pcorr = (
            torch.nan_to_num(val_metrics["PearsonCorrCoef"], nan=0.0)
            .mean()
            .item()
        )
        val_metrics["PearsonCorrCoef"] = val_mean_pcorr

        logger.info(f"Validation metrics: {val_metrics}")

        wandb.log({"finetuning/valid_loss": valid_loss, "epoch": epoch})
        wandb.log({"finetuning/valid_metrics": val_metrics, "epoch": epoch})

        if val_mean_pcorr > best_pcorr:
            best_pcorr = val_mean_pcorr
            torch.save(
                model.state_dict(), results_dir / "models" / "ft_best.pth"
            )
            # save separately the encoder and predictor
            torch.save(
                model.encoder.state_dict(),
                results_dir / "models" / "ft_best_encoder.pth",
            )
            torch.save(
                model.predictor.state_dict(),
                results_dir / "models" / "ft_best_predictor.pth",
            )
            logger.info(f"Best model saved at epoch {epoch + 1}")

        if config.training.early_stopping:
            # We use -val_mean_pcorr to maximize the Pearson correlation
            if early_stopping(-val_mean_pcorr):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        time_elapsed = time() - start_epoch_time
        logger.info(f"Epoch time: {time_elapsed:.2f} seconds")

    total_time = time() - start_training_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

    wandb.log({"finetuning/total_time": total_time})

    torch.save(model.state_dict(), results_dir / "models" / "ft_final.pth")

    visualize_loss(train_history, results_dir, filename="ft_loss.png")

    wandb.finish()


if __name__ == "__main__":

    finetune()
