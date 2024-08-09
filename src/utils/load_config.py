import yaml
from data_models.config_models import (
    Config,
    DataConfig,
    NNConfig,
    TrainingConfig,
    TestingConfig,
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

    data_conf = DataConfig(**config["DATA"])
    encoder_conf = NNConfig(**config["TRAINING"]["ENCODER"])
    predictor_conf = NNConfig(**config["TRAINING"]["PREDICTOR"])
    training_conf = TrainingConfig(
        encoder=encoder_conf,
        predictor=predictor_conf,
        batch_size=config["TRAINING"]["batch_size"],
        epochs=config["TRAINING"]["epochs"],
        num_units=config["TRAINING"]["num_units"],
        early_stopping=config["TRAINING"]["early_stopping"],
        early_stopping_patience=config["TRAINING"]["early_stopping_patience"],
        save_logs=config["TRAINING"]["save_logs"],
    )
    testing_config = TestingConfig(
        batch_size=config["TESTING"]["batch_size"],
        weights=config["TESTING"]["weights"],
        metrics=config["TESTING"]["metrics"],
        save_logs=config["TESTING"]["save_logs"],
    )

    return Config(data=data_conf, training=training_conf, testing=testing_config)
