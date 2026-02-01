"""
Fixtures para testes de modelos.
"""
import os
import pytest
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.models.lstm_model import LSTMPredictor


@pytest.fixture
def trained_model_files(tmp_path):
    """
    Cria arquivos de modelo (.pt) e scaler (.joblib) treinados
    para uso em testes de predição.
    """
    # Criar e salvar modelo
    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    model_path = str(tmp_path / "test_model.pt")
    torch.save(model.state_dict(), model_path)

    # Criar e salvar scaler fitted
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[50.0], [200.0]]))
    scaler_path = str(tmp_path / "test_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path
