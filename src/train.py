from tqdm import tqdm
from utils import get_training_arguments, load_config, EarlyStopping
from data_handlers import H5Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from time import time
from pathlib import Path
from utils.training_utils import train_epoch, valid_epoch
from utils.logger import get_logger
from utils.file_manager import organize_folders, copy_config
from autoencoder.custom_autoencoder import CustomAutoencoder
from torchinfo import summary
from visualize.visualize_loss import visualize_loss


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

    image_shape = tuple(config.data.img_size)
    # Initialize model
    latent_dim = 100
    model = CustomAutoencoder(
        image_shape=image_shape,
        latent_dim=latent_dim,
        out_channels=16,
        activation=nn.Sigmoid(),
    )

    logger.info("Model initialized")
    summary(model, input_size=(2, *image_shape), depth=5)

    # load the datasets
    train_dataset = H5Dataset(
        path=Path(config.data.path),
        is_train=True,
        is_rgb=config.data.rgb,
    )

    # Get sample data to check dimensions
    X = train_dataset[0]
    logger.info(f"Input data point shape: {X.shape}")

    TORCH_SEED = 12
    TRAIN_SIZE = 0.8
    # Split train dataset into train and validation
    logger.info(f"Manually set PyTorch seed: {TORCH_SEED}")
    torch.manual_seed(TORCH_SEED)
    train_size = int(TRAIN_SIZE * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Define training parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    N_EPOCHS = config.training.epochs
    ENCODER_LR = config.training.encoder.learning_rate
    PREDICTOR_LR = config.training.predictor.learning_rate
    BATCH_SIZE = config.training.batch_size

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.decoder.parameters(), "lr": PREDICTOR_LR},
        ]
    )
    loss_fn = nn.MSELoss()
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

        # validation
        valid_loss = valid_epoch(
            model=model,
            valid_loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch + 1,
            results_dir=results_dir_path,
        )
        train_history["valid_loss"].append(valid_loss)

        logger.info(f"Epoch: {epoch + 1}/{N_EPOCHS} \t Validation Loss: {valid_loss}")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), results_dir_path / "models" / "best.pth")
            torch.save(
                model.encoder.state_dict(),
                results_dir_path / "models" / "encoder_best.pth",
            )
            logger.info(f"Best model saved at epoch {epoch + 1}")

        if config.training.early_stopping:
            if early_stopping(valid_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        time_elapsed = time() - start_epoch_time
        logger.info(f"Epoch time: {time_elapsed:.2f} seconds")

    total_time = time() - start_training_time
    logger.info(f"Total training time: {total_time:.2f} seconds")
    torch.save(model.state_dict(), results_dir_path / "models" / "final.pth")
    visualize_loss(train_history, results_dir_path)
