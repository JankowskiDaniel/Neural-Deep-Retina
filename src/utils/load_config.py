import yaml
from data_models.config_models import (
    Config,
    DataConfig,
    NNConfig,
    TrainingConfig
    )

def load_config(path: str) -> Config:
    """Load config from yaml file.

    Args:
        path (str): path to yaml file

    Returns:
        Config: config object
    """
    with open(path) as file:
        config = yaml.safe_load(file)
    
    data_conf = DataConfig(**config['DATA'])
    encoder_conf = NNConfig(**config["TRAINING"]['ENCODER'])
    model_conf = NNConfig(**config["TRAINING"]['MODEL'])
    training_conf = TrainingConfig(
        encoder=encoder_conf,
        model=model_conf,
        batch_size=config["TRAINING"]['batch_size'],
        epochs=config["TRAINING"]['epochs'],
        num_units=config["TRAINING"]['num_units']
    )

    return Config(data=data_conf, training=training_conf)