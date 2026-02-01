"""
Testes para o preditor de ações.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.predictor import StockPredictor, ModelNotLoadedError


def test_predict_returns_list(trained_model_files):
    """predict() deve retornar lista de floats."""
    model_path, scaler_path = trained_model_files
    predictor = StockPredictor(model_path, scaler_path)

    # Dados dummy com valores no range do scaler
    df = pd.DataFrame({"Close": np.random.rand(100) * 100 + 50})

    predictions = predictor.predict(df, days_ahead=3)

    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert all(isinstance(p, float) for p in predictions)


def test_predict_raises_when_not_loaded():
    """Deve levantar exceção se modelo não carregado."""
    predictor = StockPredictor()
    df = pd.DataFrame({"Close": [100, 101, 102]})

    with pytest.raises(ModelNotLoadedError):
        predictor.predict(df)


def test_is_loaded():
    """is_loaded deve refletir estado."""
    predictor = StockPredictor()
    assert predictor.is_loaded() is False


def test_is_loaded_after_load(trained_model_files):
    """is_loaded deve ser True apos carregar modelo."""
    model_path, scaler_path = trained_model_files
    predictor = StockPredictor(model_path, scaler_path)
    assert predictor.is_loaded() is True


def test_reload_model(trained_model_files):
    """Hot reload funciona."""
    model_path, scaler_path = trained_model_files

    predictor = StockPredictor()
    assert predictor.is_loaded() is False

    # Carregar modelo
    predictor.reload_model(model_path, scaler_path)
    assert predictor.is_loaded() is True

    # Recarregar (hot reload)
    predictor.reload_model(model_path, scaler_path)
    assert predictor.is_loaded() is True
