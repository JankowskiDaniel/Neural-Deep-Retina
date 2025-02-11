from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from time import time
from sklearn.preprocessing import MinMaxScaler
from utils.training_utils import train_epoch, valid_epoch
from utils.logger import get_logger
from utils.file_manager import organize_folders, copy_config
from utils import (
    get_training_arguments,
    load_config,
    load_model,
    EarlyStopping,
    load_data_handler,
)
from visualize.visualize_loss import visualize_loss
import wandb
from uuid import uuid4


if __name__ == "__main__":

    config_path, results_dir = get_training_arguments()
    config = load_config(config_path)

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

    logger.info(f"Predictions will be made for channels: {train_dataset.pred_channels}")
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
            "num_units": 9,
        },
        resume="allow",
    )

    # log the data length
    wandb.log(
        {"train_data_length": len(train_dataset), "val_data_length": len(val_dataset)}
    )

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

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.predictor.parameters(), "lr": PREDICTOR_LR},
        ]
    )
    wandb.config.update({"optimizer": optimizer.__class__.__name__})
    
    loss_fn = nn.L1Loss()
    wandb.config.update({"loss_fn": loss_fn.__class__.__name__})
    train_history: dict = {"train_loss": [], "valid_loss": []}

    if config.training.early_stopping:
        PATIENCE = config.training.early_stopping_patience
        early_stopping = EarlyStopping(patience=PATIENCE)

    # ########### START TRAINING ###########
    logger.info(f"Training on {DEVICE} using device: {DEVICE_NAME}")
    model.to(DEVICE)
    start_training_time = time()

    best_val_loss = float("inf")
    for epoch in tqdm(range(N_EPOCHS)):
        start_epoch_time = time()
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

        logger.info(f"Epoch: {epoch + 1}/{N_EPOCHS} \t Train Loss: {train_loss}")
        (f"Epoch: {epoch + 1}/{N_EPOCHS} \t Train Loss: {train_loss}")

        wandb.log({"training/train_loss": train_loss, "epoch": epoch})

        # validation
        valid_loss = valid_epoch(
            model=model,
            valid_loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
        )
        train_history["valid_loss"].append(valid_loss)

        logger.info(f"Epoch: {epoch + 1}/{N_EPOCHS} \t Validation Loss: {valid_loss}")

        wandb.log({"training/valid_loss": valid_loss, "epoch": epoch})

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), results_dir_path / "models" / "best.pth")
            logger.info(f"Best model saved at epoch {epoch + 1}")

        if config.training.early_stopping:
            if early_stopping(valid_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        time_elapsed = time() - start_epoch_time
        logger.info(f"Epoch time: {time_elapsed:.2f} seconds")

    total_time = time() - start_training_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

    wandb.log({"training/total_time": total_time})

    torch.save(model.state_dict(), results_dir_path / "models" / "final.pth")
    visualize_loss(train_history, results_dir_path)
    wandb.finish()
