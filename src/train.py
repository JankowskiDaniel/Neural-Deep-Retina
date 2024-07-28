from tqdm import tqdm
from utils import get_arguments, load_config, load_model
from data_handlers import H5Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from time import time
from utils.training_utils import train_epoch, valid_epoch
from utils.logger import get_logger
from utils.file_manager import organize_folders, copy_config


if __name__ == "__main__":
    
    config_path, results_dir = get_arguments()
    config = load_config(config_path)

    # Organize folders
    organize_folders(results_dir)
    copy_config(results_dir, config_path)

    logger = get_logger(
        log_to_file=config.training.save_logs,
        log_file=f"results/{results_dir}/logs.log")

    # load the model
    logger.info("Loading model...")
    model = load_model(config)

    # load the datasets
    train_dataset = H5Dataset(
        path=config.data.path,
        response_type="firing_rate_10ms",
        train=True,
        is_rgb=config.data.rgb
    )
    test_dataset = H5Dataset(
        path=config.data.path,
        response_type="firing_rate_10ms",
        train=False,
        is_rgb=config.data.rgb
    )

    TORCH_SEED = 12
    TRAIN_SIZE = 0.8
    # Split train dataset into train and validation
    logger.info(f"Manually set PyTorch seed: {TORCH_SEED}")
    torch.manual_seed(TORCH_SEED)
    train_size = int(TRAIN_SIZE * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
        {'params': model.predictor.parameters(), 'lr': PREDICTOR_LR}
    ])
    loss_fn = nn.MSELoss()

    train_history = {
        "train_loss": [],
        "valid_loss": []
    }
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
            epoch=epoch
        )
        train_history["train_loss"].append(train_loss)

        logger.info(f"Epoch: {epoch + 1}/{N_EPOCHS} \t Train Loss: {train_loss}")
        (f"Epoch: {epoch + 1}/{N_EPOCHS} \t Train Loss: {train_loss}")

        # validation
        valid_loss = valid_epoch(
            model=model,
            valid_loader=test_loader,
            loss_fn=loss_fn,
            device=DEVICE
        )
        train_history["valid_loss"].append(valid_loss)

        logger.info(f"Epoch: {epoch + 1}/{N_EPOCHS} \t Validation Loss: {valid_loss}")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f"results/{results_dir}/models/best_model.pth")
            logger.info(f"Best model saved at epoch {epoch + 1}")

        time_elapsed = time() - start_epoch_time
        logger.info(f"Epoch time: {time_elapsed:.2f} seconds")

    total_time = time() - start_training_time
    logger.info(f"Total training time: {total_time:.2f} seconds")
    torch.save(model.state_dict(), f"results/{results_dir}/models/final_model.pth")
