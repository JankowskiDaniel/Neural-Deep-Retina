import yaml
from pathlib import Path
from data_models.config_models import (
    Config,
    DataConfig,
    EncoderConfig,
    PredictorConfig,
    TrainingConfig,
    TestingConfig,
    CurriculumSchedule,
    CurriculumStageSchedule,
)


def load_config(path: Path) -> Config:
    """Load config from yaml file.

    Args:
        path (str): path to yaml file

    Returns:
        Config: config object
    """
    with open(path) as file:
        config = yaml.safe_load(file)

    data_conf = DataConfig(**config["data"])
    encoder_conf = EncoderConfig(**config["training"]["encoder"])
    predictor_conf = PredictorConfig(**config["training"]["predictor"])
    training_conf = TrainingConfig(
        encoder=encoder_conf,
        predictor=predictor_conf,
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        num_units=config["training"]["num_units"],
        early_stopping=config["training"]["early_stopping"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        save_logs=config["training"]["save_logs"],
        is_curriculum=(
            config["training"]["is_curriculum"]
            if "is_curriculum" in config["training"]
            else False
        ),
        debug_mode=(
            config["training"]["debug_mode"]
            if "debug_mode" in config["training"]
            else False
        ),
        loss_function=config["training"]["loss_function"],
    )
    testing_config = TestingConfig(
        batch_size=config["testing"]["batch_size"],
        weights=config["testing"]["weights"],
        metrics=config["testing"]["metrics"],
        save_logs=config["testing"]["save_logs"],
        run_on_train_data=config["testing"]["run_on_train_data"],
    )

    return Config(
        data=data_conf, training=training_conf, testing=testing_config
    )


def load_curriculum_schedule(path: Path) -> CurriculumSchedule:
    with open(path) as file:
        config = yaml.safe_load(file)

    curr_config = load_stages_config(config["STAGES"])
    return curr_config


def load_stages_config(stages_params: dict) -> CurriculumSchedule:
    stages_config = dict()
    for stage in stages_params:
        stage_conf = CurriculumStageSchedule(**stages_params[stage])
        stages_config[int(stage)] = stage_conf
    return CurriculumSchedule(stages=stages_config)
