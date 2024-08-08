import argparse
from pathlib import Path


def get_arguments() -> tuple[Path, Path]:
    """Read parsed arguments from command line.

    Returns:
        tuple[str, str]: Return a tuple of config path and results directory
    """
    parser = argparse.ArgumentParser(description="Main training script")

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Name of a results directory",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    return config_path, results_dir
