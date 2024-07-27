from utils import get_arguments, load_config, load_model
from data_handlers import H5Dataset
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    
    config_path, results_dir = get_arguments()
    config = load_config(config_path)

    # load the model
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

    # create the dataloader
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

    # Define training parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_EPOCHS = config.training.epochs
    ENCODER_LR = config.training.encoder.learning_rate
    PREDICTOR_LR = config.training.predictor.learning_rate

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
        {'params': model.predictor.parameters(), 'lr': PREDICTOR_LR}
    ])
