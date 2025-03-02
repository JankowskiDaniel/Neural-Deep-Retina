import os
import shutil


def organize_folders(results_dir: str) -> None:
    """Organizes folders for saving results.

    Args:
        results_dir (str): Name of the directory
    """
    # Define the base directory
    base_dir = os.path.join("results", results_dir)

    # List of directories to create
    directories_to_create = [
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "testset_predictions"),
        os.path.join(base_dir, "trainset_predictions"),
        os.path.join(base_dir, "plots"),
    ]

    # Remove existing directories if they exist
    for directory in directories_to_create:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Create directories
    for directory in directories_to_create:
        os.makedirs(directory)

    file_list = [
        f
        for f in os.listdir(base_dir)
        if os.path.isfile(os.path.join(base_dir, f))
    ]
    # Delete each file
    for file_name in file_list:
        file_path = os.path.join(base_dir, file_name)
        os.remove(file_path)


def copy_config(results_dir: str, config_path: str) -> None:
    """Copies training config file to the directory with results.

    Args:
        results_dir (str): Name of results directory
        config_path (str): Config file path
    """
    # The config file is renamed to config.yaml
    base_dir = os.path.join("results", results_dir, "config.yaml")
    shutil.copy2(config_path, base_dir)
