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

    data_conf = DataConfig(**config["DATA"])
    encoder_conf = EncoderConfig(**config["TRAINING"]["ENCODER"])
    predictor_conf = PredictorConfig(**config["TRAINING"]["PREDICTOR"])
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
        run_on_train_data=config["TESTING"]["run_on_train_data"],
    )

    return Config(data=data_conf, training=training_conf, testing=testing_config)


def load_curriculum_schedule(path: Path) -> CurriculumSchedule:
    print(path)
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
