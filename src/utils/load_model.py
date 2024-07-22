from interfaces.encoder import Encoder
from data_models.config_models import Config
from architectures.dummy import DummyCNN
from encoders import VGG16Encoder
import torch.nn as nn

ARCHITECTURES: dict[str, nn.Module] = {
    "DummyCNN": DummyCNN
}

ENCODERS: dict[str, Encoder] = {
    "VGG16Encoder": VGG16Encoder
}


def load_model(config: Config) -> nn.Module:
    enc_name: str = config.training.encoder.name
    arch_name: str = config.training.model.name

    # resolve input size
    img_size = config.data.img_size
    batch_size = config.training.batch_size
    input_size = (batch_size, *img_size)

    # initialize encoder
    encoder: Encoder = ENCODERS[enc_name](input_size=input_size)

    # initialize model
    model: nn.Module = ARCHITECTURES[arch_name](
        encoder,
        config.data.num_classes
        )
    
    return model

