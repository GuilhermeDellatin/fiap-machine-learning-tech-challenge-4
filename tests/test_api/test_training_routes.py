"""
Testes para rotas de treinamento.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_start_training_returns_202(api_client):
    """POST /training/start retorna 202."""
    with (
        patch("src.api.routes.training.get_collector") as mock_collector,
        patch("src.api.routes.training.train_model_task") as mock_train,
    ):
        mock_instance = MagicMock()
        mock_instance.validate_ticker.return_value = True
        mock_collector.return_value = mock_instance

        response = api_client.post(
            "/api/v1/training/start",
            json={"ticker": "AAPL", "epochs": 5},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "accepted"
        assert data["ticker"] == "AAPL"
        mock_train.assert_called_once()


def test_start_training_invalid_ticker_returns_404(api_client):
    """POST /training/start com ticker invalido retorna 404."""
    with patch("src.api.routes.training.get_collector") as mock_collector:
        mock_instance = MagicMock()
        mock_instance.validate_ticker.return_value = False
        mock_collector.return_value = mock_instance

        response = api_client.post(
            "/api/v1/training/start",
            json={"ticker": "INVALID", "epochs": 5},
        )
        assert response.status_code == 404


def test_get_training_status_not_found(api_client):
    """Job inexistente retorna 404."""
    response = api_client.get("/api/v1/training/status/invalid-id")
    assert response.status_code == 404


def test_get_training_status_found(api_client, test_db):
    """Job existente retorna status correto."""
    from src.database.repository import TrainingJobRepository

    repo = TrainingJobRepository()
    repo.create_job(test_db, "test-job-123", "AAPL", 100, {"lr": 0.001})

    response = api_client.get("/api/v1/training/status/test-job-123")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["ticker"] == "AAPL"
    assert data["status"] == "pending"


def test_list_jobs_empty(api_client):
    """Lista vazia quando nao ha jobs."""
    response = api_client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["jobs"] == []
    assert data["total"] == 0


def test_list_jobs_with_data(api_client, test_db):
    """Lista jobs existentes."""
    from src.database.repository import TrainingJobRepository

    repo = TrainingJobRepository()
    repo.create_job(test_db, "job-1", "AAPL", 50, {})
    repo.create_job(test_db, "job-2", "PETR4.SA", 100, {})

    response = api_client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_list_models_empty(api_client):
    """Lista vazia quando nao ha modelos."""
    response = api_client.get("/api/v1/training/models")
    assert response.status_code == 200
    assert response.json()["models"] == []


def test_list_models_with_data(api_client, test_db):
    """Lista modelos registrados."""
    from src.database.repository import ModelRegistryRepository

    repo = ModelRegistryRepository()
    repo.register_model(
        test_db,
        ticker="AAPL",
        model_path="models/aapl.pt",
        scaler_path="models/aapl_scaler.joblib",
        metrics={"mae": 0.5},
        hyperparams={"hidden_size": 64},
        epochs=100,
    )

    response = api_client.get("/api/v1/training/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 1
    assert models[0]["ticker"] == "AAPL"


def test_activate_model_not_found(api_client):
    """Ativar modelo inexistente retorna 404."""
    response = api_client.post("/api/v1/training/activate/nonexistent_version")
    assert response.status_code == 404


def test_activate_model_success(api_client, test_db):
    """Ativar modelo existente funciona."""
    from src.database.repository import ModelRegistryRepository

    repo = ModelRegistryRepository()
    model = repo.register_model(
        test_db,
        ticker="AAPL",
        model_path="models/aapl.pt",
        scaler_path="models/aapl_scaler.joblib",
        metrics={},
        hyperparams={},
        epochs=50,
    )

    response = api_client.post(f"/api/v1/training/activate/{model.version_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "activated"
    assert data["version_id"] == model.version_id
