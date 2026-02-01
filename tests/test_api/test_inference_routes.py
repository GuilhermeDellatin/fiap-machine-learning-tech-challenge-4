"""
Testes para rotas de inferencia.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_inference_no_model_returns_404(api_client):
    """Inferencia sem modelo retorna 404."""
    response = api_client.post(
        "/api/v1/inference",
        json={"ticker": "XXXX", "raw_prices": [1, 2, 3]},
    )
    assert response.status_code == 404


def test_inference_validation_error(api_client):
    """Inferencia sem data nem raw_prices retorna 422."""
    response = api_client.post(
        "/api/v1/inference",
        json={"ticker": "AAPL"},
    )
    assert response.status_code == 422


def test_inference_both_data_and_prices_error(api_client):
    """Fornecer data e raw_prices ao mesmo tempo retorna 422."""
    response = api_client.post(
        "/api/v1/inference",
        json={
            "ticker": "AAPL",
            "data": [[1.0]],
            "raw_prices": [1.0],
        },
    )
    assert response.status_code == 422


def test_warmup_no_model(api_client):
    """Warmup sem modelo carregado."""
    with patch("src.api.routes.inference.get_predictor") as mock_pred_fn:
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False
        mock_pred_fn.return_value = mock_predictor

        response = api_client.get("/api/v1/inference/warmup")
        assert response.status_code == 200
        assert response.json()["status"] == "no_model_loaded"


def test_warmup_with_model(api_client):
    """Warmup com modelo carregado."""
    import torch

    with patch("src.api.routes.inference.get_predictor") as mock_pred_fn:
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.device = torch.device("cpu")
        mock_predictor.inference.return_value = torch.tensor([[0.5]])
        mock_pred_fn.return_value = mock_predictor

        response = api_client.get("/api/v1/inference/warmup")
        assert response.status_code == 200
        assert response.json()["status"] == "warmed_up"
        assert response.json()["device"] == "cpu"


def test_inference_with_raw_prices(api_client, test_db, trained_model_files):
    """Inferencia com raw_prices e modelo ativo."""
    from src.database.repository import ModelRegistryRepository

    model_path, scaler_path = trained_model_files

    repo = ModelRegistryRepository()
    model = repo.register_model(
        test_db,
        ticker="AAPL",
        model_path=model_path,
        scaler_path=scaler_path,
        metrics={},
        hyperparams={},
        epochs=50,
    )
    repo.set_active_model(test_db, "AAPL", model.version_id)

    import numpy as np

    raw_prices = (np.random.rand(100) * 100 + 50).tolist()

    with patch("src.api.routes.inference.get_predictor") as mock_pred_fn:
        from src.models.predictor import StockPredictor

        predictor = StockPredictor(model_path, scaler_path)
        predictor.current_ticker = "AAPL"
        predictor.model_version = model.version_id
        mock_pred_fn.return_value = predictor

        response = api_client.post(
            "/api/v1/inference",
            json={"ticker": "AAPL", "raw_prices": raw_prices},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert isinstance(data["prediction"], float)
    assert data["is_normalized"] is False
    assert data["inference_time_ms"] >= 0


def test_batch_inference_no_model(api_client):
    """Batch inference sem modelo retorna 404."""
    response = api_client.post(
        "/api/v1/inference/batch",
        json={
            "ticker": "XXXX",
            "sequences": [[[1.0], [2.0]]],
        },
    )
    assert response.status_code == 404
