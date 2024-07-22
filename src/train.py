from utils import get_arguments, load_config, load_model


if __name__ == "__main__":
    
    config_path, results_dir = get_arguments()
    config = load_config(config_path)

    # load the model
    model = load_model(config)

