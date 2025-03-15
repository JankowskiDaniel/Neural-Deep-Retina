from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from time import time
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.regression import PearsonCorrCoef
from utils.training_utils import train_epoch, valid_epoch
from utils.logger import get_logger
from utils.file_manager import organize_folders, copy_config
from data_handlers import (
    CurriculumHandler,
    CurriculumDatasets,
    CurriculumDataloaders,
)
from utils import (
    get_training_arguments,
    get_metric_tracker,
    load_config,
    load_curriculum_schedule,
    load_model,
    EarlyStopping,
    load_data_handler,
)
from visualize.visualize_loss import visualize_loss
import wandb
from uuid import uuid4
from torchinfo import summary


if __name__ == "__main__":

    config_path, curriculum_schedule_path, results_dir, if_wandb = (
        get_training_arguments()
    )
    print(if_wandb)
    config = load_config(config_path)

    curr_schedule = None
    if config.training.is_curriculum:
        curr_schedule = load_curriculum_schedule(curriculum_schedule_path)

    # Organize folders
    organize_folders(results_dir)
    copy_config(results_dir, config_path)

    # Create path object to results directory
    results_dir_path = "results" / results_dir

    logger = get_logger(
        log_to_file=config.training.save_logs,
        log_file=results_dir_path / "train_logs.log",
    )

    # load the model
    if not if_wandb:
        logger.warning("Logging to wandb is disabled.")
    logger.info("Loading model...")
    model = load_model(config)

    y_scaler = MinMaxScaler()  # StandardScaler(with_mean=False)

    # load the data handler
    logger.info("Loading data handler...")
    train_dataset = load_data_handler(
        config.data,
        results_dir=results_dir_path,
        is_train=True,
        y_scaler=y_scaler,
    )
    val_dataset = load_data_handler(
        config.data,
        results_dir=results_dir_path,
        is_train=False,
        subset_type="val",
        use_saved_scaler=True,
        y_scaler=y_scaler,
    )

    # Get sample data to check dimensions
    X, y = train_dataset[0]
    logger.info(f"Input data point shape: {X.shape}")
    logger.info(f"Target data point shape: {y.shape}")

    logger.info(
        f"Predictions will be made for channels: {train_dataset.pred_channels}"
    )
    TORCH_SEED = 12
    TRAIN_SIZE = 0.8
    # Split train dataset into train and validation
    logger.info(f"Manually set PyTorch seed: {TORCH_SEED}")
    torch.manual_seed(TORCH_SEED)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Define training parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = False
    NUM_WORKERS = 0
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    N_EPOCHS = config.training.epochs
    ENCODER_LR = config.training.encoder.learning_rate
    PREDICTOR_LR = config.training.predictor.learning_rate
    BATCH_SIZE = config.training.batch_size

    _id = str(uuid4())

    # save _id into txt file
    with open(results_dir_path / "id.txt", "w") as f:
        f.write(_id)

    if if_wandb:
        wandb.init(
            entity="jankowskidaniel06-put",
            project="Neural Deep Retina",
            name=str(results_dir_path),
            id=_id,
            config={
                "data": {
                    "data_handler": config.data.data_handler,
                    "img_shape": config.data.img_shape,
                    "is_rgb": config.data.is_rgb,
                    "seq_len": config.data.seq_len,
                    "prediction_step": config.data.prediction_step,
                    "scaler": y_scaler.__class__.__name__,
                    "prediction_channels": config.data.pred_channels,
                    "is_classification": config.data.is_classification,
                    "class_epsilon": config.data.class_epsilon,
                    "is_curriculum": config.training.is_curriculum,
                },
                "model": {
                    "encoder": {
                        "name": config.training.encoder.name,
                        "freeze": config.training.encoder.freeze,
                        "learning_rate": config.training.encoder.learning_rate,
                        "n_trainable_params": model.encoder_n_trainable_params,
                    },
                    "predictor": {
                        "name": config.training.predictor.name,
                        "learning_rate": config.training.predictor.learning_rate,
                        "n_trainable_params": model.predictor_n_trainable_params,
                    },
                    "total_trainable_params": model.total_n_trainable_params,
                },
                "epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "num_units": config.training.num_units,
            },
            resume="allow",
        )

    # log this additionaly to easily filter classification runs in the main
    # table in wandb
    if if_wandb:
        wandb.log({"is_classification": config.data.is_classification})

        # Log the data length
        wandb.log(
            {
                "train_data_length": len(train_dataset),
                "val_data_length": len(val_dataset),
            }
        )

    # Get model summary
    model_summary_str = str(
        summary(model, model.input_shape, device=DEVICE, verbose=0)
    )
    # Save to a txt file
    model_summary_filename = results_dir_path / "model_summary.txt"
    with open(model_summary_filename, "w", encoding="utf-8") as f:
        f.write(model_summary_str)
    # Log model summary txt to wandb
    if if_wandb:
        model_summary_artifact = wandb.Artifact(
            "model_summary", "model_details"
        )
        model_summary_artifact.add_file(model_summary_filename)
        wandb.log_artifact(model_summary_artifact)

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
        curriculum_schedule=curr_schedule,
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
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=3,
        cooldown=2,
        threshold=0.01,
    )

    if if_wandb:
        wandb.config.update({"optimizer": optimizer.__class__.__name__})

    # get Y to compute pos_weight for BCEWithLogitsLoss
    Y = train_dataset.get_target()

    # pos_counts = np.sum(Y, axis=1)  # Count of positive (1s) per class
    # neg_counts = Y.shape[1] - pos_counts  # Count of negative (0s) per class

    # # Compute pos_weight (negatives / positives), ensuring no division by zero
    # pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)

    # # Convert to PyTorch tensor
    # pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)
    # print(f"POS WEIGHT: {pos_weight_tensor}")
    loss_fn = nn.MSELoss()
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if if_wandb:
        wandb.config.update({"loss_fn": loss_fn.__class__.__name__})
    train_history: dict = {"train_loss": [], "valid_loss": []}

    if config.training.early_stopping:
        PATIENCE = config.training.early_stopping_patience
        early_stopping = EarlyStopping(patience=PATIENCE)

    val_metric_tracker = get_metric_tracker(
        ["mae", "mse", "pcorr"],
        initialized_metrics=[
            PearsonCorrCoef(num_outputs=config.training.num_units).to(DEVICE)
        ],
        DEVICE=DEVICE,
    )

    # ########### START TRAINING ###########
    logger.info(f"Training on {DEVICE} using device: {DEVICE_NAME}")
    model.to(DEVICE)
    start_training_time = time()

    best_pcorr = 0.0
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

        if if_wandb:
            wandb.log({"training/train_loss": train_loss, "epoch": epoch})

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
        scheduler.step(valid_loss)
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

        if if_wandb:
            wandb.log({"training/valid_loss": valid_loss, "epoch": epoch})
            wandb.log({"training/valid_metrics": val_metrics, "epoch": epoch})

        if val_mean_pcorr > best_pcorr:
            best_pcorr = val_mean_pcorr
            torch.save(
                model.state_dict(), results_dir_path / "models" / "best.pth"
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

    if if_wandb:
        wandb.log({"training/total_time": total_time})

    torch.save(model.state_dict(), results_dir_path / "models" / "final.pth")
    visualize_loss(train_history, results_dir_path)

    if if_wandb:
        wandb.finish()
