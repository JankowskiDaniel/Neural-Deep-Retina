from pathlib import Path
from interfaces import Encoder, Predictor
from data_models.config_models import Config
from predictors.linears import SingleLinear, MultiLinear
from predictors.rnns import SingleLSTM
from encoders import VGG16Encoder, MC3VideoEncoder, CustomEncoder
from models import DeepRetinaModel

PREDICTORS: dict[str, Predictor] = {
    "SingleLinear": SingleLinear,
    "MultiLinear": MultiLinear,
    "SingleLSTM": SingleLSTM,
}

ENCODERS: dict[str, Encoder] = {
    "VGG16Encoder": VGG16Encoder,
    "MC3VideoEncoder": MC3VideoEncoder,
    "CustomEncoder": CustomEncoder,
}


def load_model(config: Config) -> DeepRetinaModel:
    enc_name: str = config.training.encoder.name
    pred_name: str = config.training.predictor.name

    # resolve input size
    img_shape = config.data.img_shape
    is_rgb = config.data.is_rgb
    if is_rgb:
        img_shape[0] = 3
    batch_size = config.training.batch_size
    seq_len = config.data.seq_len
    if seq_len > 1:
        if enc_name == "CustomEncoder":
            # CustomEncoder works with (batch, seq_len, height, width)
            input_shape = (batch_size, seq_len, *img_shape[1:])
        else:
            input_shape = (batch_size, seq_len, *img_shape)
    else:
        input_shape = (batch_size, *img_shape)

    # Resolve encoder weights
    weights_path = Path("pretrained_weights") / config.training.encoder.weights
    # Check if encoder weights exist
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights at {weights_path}")
    freeze = config.training.encoder.freeze

    # initialize encoder
    encoder: Encoder = ENCODERS[enc_name](
        input_shape=input_shape,
        weights_path=weights_path,
        freeze=freeze,
        seq_len=seq_len,
    )
    encoder_output_shape = encoder.get_output_shape()

    # initialize predictor
    predictor: Predictor = PREDICTORS[pred_name](
        input_size=encoder_output_shape, num_classes=config.training.num_units
    )

    model = DeepRetinaModel(encoder=encoder, predictor=predictor)

    return model
