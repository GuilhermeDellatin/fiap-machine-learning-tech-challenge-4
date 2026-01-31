"""
Testes para os repositórios do banco de dados.
"""
import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.models import PriceCache, TrainingJob, ModelRegistry
from src.database.repository import (
    PriceCacheRepository,
    TrainingJobRepository,
    ModelRegistryRepository,
)


@pytest.fixture
def db_session():
    """Cria sessão de banco de dados em memória para testes."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def price_repo():
    return PriceCacheRepository()


@pytest.fixture
def job_repo():
    return TrainingJobRepository()


@pytest.fixture
def model_repo():
    return ModelRegistryRepository()


@pytest.fixture
def sample_df():
    """DataFrame de exemplo simulando dados do yfinance."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0, 10.8, 11.2],
            "High": [10.5, 11.0, 11.5, 11.0, 11.5],
            "Low": [9.5, 10.0, 10.5, 10.5, 11.0],
            "Close": [10.2, 10.8, 11.2, 10.9, 11.3],
            "Adj Close": [10.2, 10.8, 11.2, 10.9, 11.3],
            "Volume": [1000, 1100, 1200, 900, 1300],
        },
        index=dates,
    )


# --- PriceCacheRepository Tests ---


def test_save_and_get_prices(db_session, price_repo, sample_df):
    """Salvar preços e recuperar do cache."""
    count = price_repo.save_prices(db_session, "PETR4.SA", sample_df)
    assert count == 5

    prices = price_repo.get_prices(
        db_session, "PETR4.SA", date(2024, 1, 1), date(2024, 1, 31)
    )
    assert len(prices) == 5
    assert prices[0].ticker == "PETR4.SA"
    assert prices[0].close == 10.2


def test_cache_valid_when_fresh(db_session, price_repo, sample_df):
    """Cache com updated_at recente deve ser válido."""
    price_repo.save_prices(db_session, "PETR4.SA", sample_df)

    valid = price_repo.is_cache_valid(
        db_session, "PETR4.SA", date(2024, 1, 1), date(2024, 1, 31)
    )
    assert valid is True


def test_cache_invalid_when_expired(db_session, price_repo, sample_df):
    """Cache com updated_at > 24h deve ser inválido."""
    price_repo.save_prices(db_session, "PETR4.SA", sample_df)

    # Forçar updated_at para 25 horas atrás
    old_time = datetime.utcnow() - timedelta(hours=25)
    db_session.query(PriceCache).update({"updated_at": old_time})
    db_session.commit()

    valid = price_repo.is_cache_valid(
        db_session, "PETR4.SA", date(2024, 1, 1), date(2024, 1, 31)
    )
    assert valid is False


def test_cache_invalid_when_empty(db_session, price_repo):
    """Cache inexistente deve ser inválido."""
    valid = price_repo.is_cache_valid(
        db_session, "VALE3.SA", date(2024, 1, 1), date(2024, 1, 31)
    )
    assert valid is False


# --- TrainingJobRepository Tests ---


def test_create_training_job(db_session, job_repo):
    """Criar job com status pending."""
    job = job_repo.create_job(
        db_session,
        job_id="test-uuid-001",
        ticker="PETR4.SA",
        epochs=100,
        hyperparams={"lr": 0.001, "hidden_size": 64},
    )
    assert job.job_id == "test-uuid-001"
    assert job.ticker == "PETR4.SA"
    assert job.status == "pending"
    assert job.epochs_total == 100


def test_update_training_job_status(db_session, job_repo):
    """Atualizar status do job."""
    job_repo.create_job(
        db_session,
        job_id="test-uuid-002",
        ticker="PETR4.SA",
        epochs=100,
        hyperparams={},
    )

    updated = job_repo.update_job(
        db_session,
        "test-uuid-002",
        status="running",
        started_at=datetime.utcnow(),
    )
    assert updated is not None
    assert updated.status == "running"
    assert updated.started_at is not None


# --- ModelRegistryRepository Tests ---


def test_register_model(db_session, model_repo):
    """Registrar modelo com version_id gerado."""
    model = model_repo.register_model(
        db_session,
        ticker="PETR4.SA",
        model_path="models/PETR4.SA_20240101.pt",
        scaler_path="models/PETR4.SA_20240101_scaler.joblib",
        metrics={"mae": 0.5, "rmse": 0.7, "mape": 2.1, "r2_score": 0.95},
        hyperparams={"hidden_size": 64, "num_layers": 2},
        epochs=100,
    )
    assert model.version_id.startswith("PETR4.SA_")
    assert model.ticker == "PETR4.SA"
    assert model.mae == 0.5
    assert model.is_active is False


def test_set_active_model(db_session, model_repo):
    """Ativar modelo deve desativar outros do mesmo ticker."""
    m1 = model_repo.register_model(
        db_session,
        ticker="PETR4.SA",
        model_path="models/m1.pt",
        scaler_path="models/m1_scaler.joblib",
        metrics={},
        hyperparams={},
        epochs=50,
    )

    import time
    time.sleep(1)  # Garantir version_id diferente

    m2 = model_repo.register_model(
        db_session,
        ticker="PETR4.SA",
        model_path="models/m2.pt",
        scaler_path="models/m2_scaler.joblib",
        metrics={},
        hyperparams={},
        epochs=100,
    )

    # Ativar primeiro modelo
    model_repo.set_active_model(db_session, "PETR4.SA", m1.version_id)
    assert model_repo.get_active_model(db_session, "PETR4.SA").version_id == m1.version_id

    # Ativar segundo modelo - primeiro deve ser desativado
    model_repo.set_active_model(db_session, "PETR4.SA", m2.version_id)
    active = model_repo.get_active_model(db_session, "PETR4.SA")
    assert active.version_id == m2.version_id

    # Verificar que apenas 1 está ativo
    active_models = model_repo.list_models(db_session, ticker="PETR4.SA", active_only=True)
    assert len(active_models) == 1


def test_get_active_model_returns_none(db_session, model_repo):
    """Sem modelo ativo deve retornar None."""
    result = model_repo.get_active_model(db_session, "VALE3.SA")
    assert result is None
