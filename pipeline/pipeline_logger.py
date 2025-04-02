from pathlib import Path
import logging


def setup_step_logger(
    log_dir: Path, step_name: str, level=logging.INFO
) -> logging.Logger:
    """ "Set up a logger for a specific step in the pipeline.
    Args:
        log_dir (Path): Directory where logs will be stored.
        step_name (str): Name of the step to be logged.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger for the step.
    """
    # Make the log file for the step
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{step_name}.log"

    logger = logging.getLogger(f"pipeline.{step_name}")
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)

    logger.info(f"Logger initialized for step '{step_name}' at {log_file}")
    return logger
