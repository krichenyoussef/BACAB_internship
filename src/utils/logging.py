import logging
import sys

_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


def create_logger(log_level: str = "INFO") -> logging.Logger:
    if log_level not in _LEVELS:
        raise ValueError(f"Invalid LOG_LEVEL {log_level!r}, expected one of {list(_LEVELS)}")
    logger = logging.getLogger("src")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(_LEVELS[log_level])
    return logger
