from interfaces import Encoder, Predictor
from data_models.config_models import Config
from predictors.dummy import DummyCNN
from encoders import VGG16Encoder
from models import DeepRetinaModel

PREDICTORS: dict[str, Predictor] = {
    "DummyCNN": DummyCNN
}

ENCODERS: dict[str, Encoder] = {
    "VGG16Encoder": VGG16Encoder
}


def load_model(config: Config) -> DeepRetinaModel:
    enc_name: str = config.training.encoder.name
    pred_name: str = config.training.predictor.name

    # resolve input size
    img_size = config.data.img_size
    is_rgb = config.data.rgb
    if is_rgb:
        img_size[0] = 3
    batch_size = config.training.batch_size
    input_size = (batch_size, *img_size)

    # initialize encoder
    encoder: Encoder = ENCODERS[enc_name](input_size=input_size)
    encoder_output_shape = encoder.get_output_shape()
    flattened_size = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]

    # initialize predictor
    predictor: Predictor = PREDICTORS[pred_name](
        input_size=flattened_size,
        num_classes=config.training.num_units
    )

    model = DeepRetinaModel(encoder=encoder, predictor=predictor)

    return model

