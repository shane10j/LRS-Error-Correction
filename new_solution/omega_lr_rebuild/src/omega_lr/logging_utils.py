"""Logging setup."""

import logging


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

