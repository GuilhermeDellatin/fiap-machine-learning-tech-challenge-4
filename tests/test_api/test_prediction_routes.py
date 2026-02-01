"""
Testes para rotas de predicao.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_predict_no_model_returns_404(api_client):
    """Predicao sem modelo retorna 404."""
    response = api_client.post(
        "/api/v1/predict",
        json={"ticker": "XXXX", "days_ahead": 3},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "suggestion" in detail
    assert "No trained model found" in detail["error"]


def test_predict_get_no_model_returns_404(api_client):
    """GET /predict/{ticker} sem modelo retorna 404."""
    response = api_client.get("/api/v1/predict/AAPL?days_ahead=1")
    assert response.status_code == 404


def test_predict_batch_all_fail(api_client):
    """Batch prediction com tickers sem modelo registra falhas."""
    response = api_client.post(
        "/api/v1/predict/batch",
        json={"tickers": ["XXXX", "YYYY"], "days_ahead": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["failed"]) == 2
    assert data["predictions"] == {}


def test_predict_with_model(api_client, test_db, trained_model_files):
    """Predicao com modelo treinado retorna 200."""
    from src.database.repository import ModelRegistryRepository

    model_path, scaler_path = trained_model_files

    # Registrar e ativar modelo
    repo = ModelRegistryRepository()
    model = repo.register_model(
        test_db,
        ticker="AAPL",
        model_path=model_path,
        scaler_path=scaler_path,
        metrics={"mae": 0.5},
        hyperparams={},
        epochs=50,
    )
    repo.set_active_model(test_db, "AAPL", model.version_id)

    # Mock do collector para retornar dados historicos
    import pandas as pd
    import numpy as np

    mock_df = pd.DataFrame({"Close": np.random.rand(100) * 100 + 50})

    with patch("src.api.routes.prediction.get_collector") as mock_coll:
        mock_instance = MagicMock()
        mock_instance.download_data.return_value = mock_df
        mock_coll.return_value = mock_instance

        # Resetar predictor global para evitar estado de outros testes
        with patch("src.api.routes.prediction.get_predictor") as mock_pred_fn:
            from src.models.predictor import StockPredictor

            predictor = StockPredictor(model_path, scaler_path)
            predictor.current_ticker = "AAPL"
            predictor.model_version = model.version_id
            mock_pred_fn.return_value = predictor

            response = api_client.post(
                "/api/v1/predict",
                json={"ticker": "AAPL", "days_ahead": 2},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert len(data["predictions"]) == 2
    assert "price" in data["predictions"][0]
    assert "date" in data["predictions"][0]
