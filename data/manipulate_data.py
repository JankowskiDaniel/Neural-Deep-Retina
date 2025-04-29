import argparse
import sys
from pathlib import Path
from logging import Logger

sys.path.append("src")
from src.utils.logger import get_logger  # noqa: E402
from data.neural_code_data_manipulator import NeuralCodeData  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument('-mt', "--man_type", type=str, choices=['make_val_split', 'add_noise'], required=True,
                        help="Specify the scenario to execute: make validation split or add noise to data")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-s", "--sigma", type=float, default=0.3)
    args = parser.parse_args()

    logger = get_logger()

    ncd = NeuralCodeData(args.path, logger)
    logger.info(f"Reading data from {args.path}")
    ncd.read_data()
    if args.man_type == "add_noise":
        logger.info(f"Adding Gaussian noise with sigma {args.sigma}")
        ncd.add_gaussian_noise(sigma=args.sigma)
    elif args.man_type == 'make_val_split':
        logger.info(f"Creating a validation split with ratio {args.train_ratio}")
        ncd.make_validation_split(args.train_ratio)
    else:
        raise ValueError(f"Unknown manipulation type {args.man_type}")
    ncd.close()
