"""
Testes para a arquitetura LSTM.
"""
import torch
from src.models.lstm_model import LSTMPredictor


def test_forward_shape():
    """Output deve ter shape correto."""
    model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=2)

    batch_size = 16
    sequence_length = 60
    x = torch.randn(batch_size, sequence_length, 1)

    output = model(x)

    assert output.shape == (batch_size, 1)


def test_different_batch_sizes():
    """Modelo deve aceitar diferentes batch sizes."""
    model = LSTMPredictor()

    for batch_size in [1, 8, 32, 64]:
        x = torch.randn(batch_size, 60, 1)
        output = model(x)
        assert output.shape == (batch_size, 1)


def test_get_hyperparameters():
    """Deve retornar dict com hiperparâmetros."""
    model = LSTMPredictor(hidden_size=128, num_layers=3)

    params = model.get_hyperparameters()

    assert params["hidden_size"] == 128
    assert params["num_layers"] == 3
