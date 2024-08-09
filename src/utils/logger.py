import logging


def get_logger(log_to_file: bool = False, log_file: str = "logs.log"):
    """
    Create and configure a logger object.
    Parameters:
        log_to_file (bool): Flag indicating whether to log messages to a file. Default is False.
        log_file (str): Path to the log file. Default is "logs.log".
    Returns:
        logger (logging.Logger): Configured logger object.
    """  # noqa: E501
    # Create a custom logger
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    if log_to_file:
        # Create a file handler in a write mode
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger
