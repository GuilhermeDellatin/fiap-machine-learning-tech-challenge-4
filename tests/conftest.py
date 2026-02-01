"""
Fixtures compartilhadas para testes.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from src.database.connection import Base
from src.database.models import PriceCache, TrainingJob, ModelRegistry
from src.api.main import app


@pytest.fixture
def test_db():
    """Banco SQLite em memoria para testes."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()

    yield db

    db.close()


@pytest.fixture
def sample_stock_data():
    """DataFrame com dados de exemplo."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.random.rand(100) * 100 + 50,
            "High": np.random.rand(100) * 100 + 55,
            "Low": np.random.rand(100) * 100 + 45,
            "Close": np.random.rand(100) * 100 + 50,
            "Adj Close": np.random.rand(100) * 100 + 50,
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
    )


@pytest.fixture
def sample_prices_in_cache(test_db):
    """Cache preenchido com dados recentes."""
    now = datetime.utcnow()
    for i in range(100):
        test_db.add(
            PriceCache(
                ticker="AAPL",
                date=datetime.now().date() - timedelta(days=100 - i),
                open=100 + i,
                high=105 + i,
                low=95 + i,
                close=102 + i,
                volume=1000000,
                created_at=now,
                updated_at=now,
            )
        )
    test_db.commit()
    return test_db


@pytest.fixture
def expired_cache(test_db):
    """Cache expirado (>24h)."""
    old_time = datetime.utcnow() - timedelta(hours=48)
    for i in range(10):
        price = PriceCache(
            ticker="AAPL",
            date=datetime.now().date() - timedelta(days=10 - i),
            open=100 + i,
            high=105 + i,
            low=95 + i,
            close=100 + i,
            volume=1000000,
            created_at=old_time,
            updated_at=old_time,
        )
        test_db.add(price)
    test_db.commit()

    return test_db


@pytest.fixture
def trained_model_files(tmp_path):
    """Cria arquivos de modelo dummy para testes."""
    import torch
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    from src.models.lstm_model import LSTMPredictor

    # Modelo
    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit([[0], [100]])
    scaler_path = tmp_path / "test_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    return str(model_path), str(scaler_path)


@pytest.fixture
def api_client(test_db):
    """TestClient com banco de teste."""
    from src.api.dependencies import get_database

    def override_get_db():
        yield test_db

    app.dependency_overrides[get_database] = override_get_db

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
