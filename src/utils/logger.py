"""
Sistema de logging estruturado.
"""
import logging
import sys
from src.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Retorna logger configurado.

    Args:
        name: Nome do módulo (use __name__)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    return logger
