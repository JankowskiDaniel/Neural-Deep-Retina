import argparse
from pathlib import Path


def get_training_arguments() -> tuple[Path, Path]:
    """Parse training arguments from command line.

    Returns:
        tuple[Path, Path]: Return a tuple of config path and results directory
    """
    parser = argparse.ArgumentParser(description="Main training script")

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Name of the results directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Name of th config file. Assumed to be in the configs directory",
    )

    args = parser.parse_args()

    config_path = Path("configs") / args.config
    results_dir = Path(args.results_dir)
    return config_path, results_dir


def get_testing_arguments() -> Path:
    """Parse testing arguments from command line.
    The config file is assumed to be in the results directory.

    Returns:
        Path: Return results directory
    """
    parser = argparse.ArgumentParser(description="Main testing script")

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Name of the results directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    return results_dir
