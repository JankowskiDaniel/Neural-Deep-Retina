import argparse
from pathlib import Path


def get_training_arguments() -> tuple[Path, Path, bool]:
    """Parse training arguments from command line.

    Returns:
        tuple[Path, Path, bool]: Return a tuple of config path, results
        directory and if wandb logging is enabled
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

    parser.add_argument(
        "--no_log_wandb",
        action="store_false",
        help=(
            "Whether to log to wandb. "
            "If True, make sure to set the WANDB_API_KEY environment variable"
        ),
    )

    args = parser.parse_args()

    config_path = Path("configs") / args.config
    results_dir = Path(args.results_dir)
    if_wandb = args.no_log_wandb
    return config_path, results_dir, if_wandb


def get_testing_arguments() -> tuple[Path, bool]:
    """Parse testing arguments from command line.
    The config file is assumed to be in the results directory.

    Returns:
        tuple[Path, bool]: Return results directory,
        and if wandb logging is enabled
    """
    parser = argparse.ArgumentParser(description="Main testing script")

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Name of the results directory",
    )

    parser.add_argument(
        "--no_log_wandb",
        action="store_false",
        help=(
            "Whether to log to wandb. "
        ),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if_wandb = args.no_log_wandb
    return results_dir, if_wandb
