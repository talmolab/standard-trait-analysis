from datetime import datetime
from pathlib import Path
import logging


def setup_logger(
    base_dir="runs", run_prefix="run", log_name="log.txt", level=logging.INFO
):
    """Creates a timestamped run directory and sets up a logger."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{run_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / log_name

    logger = logging.getLogger("marimo_logger")

    # Prevent duplicate handlers on rerun
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)

    logger.info(f"Logger initialized. Logs are being saved to: {log_path}")

    return logger, run_dir
