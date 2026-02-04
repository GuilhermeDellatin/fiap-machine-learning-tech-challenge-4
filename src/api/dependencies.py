"""
FastAPI dependencies.
"""
from functools import lru_cache
from typing import Generator
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.data.collector import StockDataCollector
from src.models.predictor import StockPredictor


def get_database() -> Generator[Session, None, None]:
    """Dependency para injetar sessão do banco."""
    yield from get_db()


@lru_cache()
def get_collector() -> StockDataCollector:
    """Singleton do collector."""
    return StockDataCollector()


_predictor_instance: StockPredictor | None = None


def get_predictor() -> StockPredictor:
    """Retorna instância do predictor (lazy loading)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = StockPredictor()
    return _predictor_instance
