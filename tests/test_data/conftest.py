"""
Fixtures compartilhadas para testes de data.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import PriceCache


@pytest.fixture
def test_db():
    """Sessão de banco em memória."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_prices_in_cache(test_db):
    """Insere preços recentes no cache (válido)."""
    now = datetime.utcnow()
    for i in range(5):
        day = datetime(2024, 1, 2 + i).date()
        record = PriceCache(
            ticker="AAPL",
            date=day,
            open=150.0 + i,
            high=155.0 + i,
            low=148.0 + i,
            close=152.0 + i,
            adj_close=152.0 + i,
            volume=1000000 + i * 1000,
            created_at=now,
            updated_at=now,
        )
        test_db.add(record)
    test_db.commit()
    return test_db


@pytest.fixture
def expired_cache(test_db):
    """Insere preços com cache expirado (>24h)."""
    old_time = datetime.utcnow() - timedelta(hours=25)
    for i in range(5):
        day = datetime(2024, 1, 2 + i).date()
        record = PriceCache(
            ticker="AAPL",
            date=day,
            open=150.0 + i,
            high=155.0 + i,
            low=148.0 + i,
            close=152.0 + i,
            adj_close=152.0 + i,
            volume=1000000 + i * 1000,
            created_at=old_time,
            updated_at=old_time,
        )
        test_db.add(record)
    test_db.commit()
    return test_db


@pytest.fixture
def sample_yfinance_df():
    """DataFrame simulando retorno do yfinance após reset_index."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Date": [d.date() for d in dates],
            "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "High": [155.0, 156.0, 157.0, 158.0, 159.0],
            "Low": [148.0, 149.0, 150.0, 151.0, 152.0],
            "Close": [152.0, 153.0, 154.0, 155.0, 156.0],
            "Adj Close": [152.0, 153.0, 154.0, 155.0, 156.0],
            "Volume": [1000000, 1001000, 1002000, 1003000, 1004000],
        }
    )
