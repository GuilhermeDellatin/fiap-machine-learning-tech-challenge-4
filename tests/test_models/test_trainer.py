"""
Testes para o treinamento do modelo LSTM.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer


@pytest.fixture
def model():
    """Modelo LSTM pequeno para testes."""
    return LSTMPredictor(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)


@pytest.fixture
def trainer(model):
    """Trainer configurado."""
    return ModelTrainer(model, learning_rate=0.01, early_stopping_patience=3)


@pytest.fixture
def dummy_data():
    """Dados sinteticos para treino/validacao/teste."""
    np.random.seed(42)
    n_samples = 50
    seq_len = 10
    X = np.random.rand(n_samples, seq_len, 1).astype(np.float32)
    y = np.random.rand(n_samples, 1).astype(np.float32)
    return X, y


def test_create_dataloader(trainer, dummy_data):
    """DataLoader criado com tamanho correto."""
    X, y = dummy_data
    loader = trainer.create_dataloader(X, y, batch_size=16)

    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape == (16, 10, 1)
    assert batch_y.shape == (16, 1)


def test_train_returns_history(trainer, dummy_data):
    """Treinamento retorna historico com losses."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    history = trainer.train(train_loader, val_loader, epochs=5)

    assert "train_losses" in history
    assert "val_losses" in history
    assert "best_epoch" in history
    assert len(history["train_losses"]) == 5
    assert len(history["val_losses"]) == 5


def test_train_with_progress_callback(trainer, dummy_data):
    """Callback de progresso e chamado a cada epoca."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    callback = MagicMock()
    trainer.train(train_loader, val_loader, epochs=3, progress_callback=callback)

    assert callback.call_count == 3


def test_train_early_stopping(dummy_data):
    """Early stopping para treinamento quando val_loss nao melhora."""
    model = LSTMPredictor(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    trainer = ModelTrainer(model, learning_rate=0.01, early_stopping_patience=2)

    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    history = trainer.train(train_loader, val_loader, epochs=100)

    # Com patience=2, deve parar antes de 100 epocas
    assert len(history["train_losses"]) < 100


def test_evaluate_returns_metrics(trainer, dummy_data):
    """Avaliacao retorna metricas esperadas."""
    X, y = dummy_data
    test_loader = trainer.create_dataloader(X, y, batch_size=16, shuffle=False)

    metrics = trainer.evaluate(test_loader)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "r2_score" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_save_and_load_checkpoint(trainer, dummy_data, tmp_path):
    """Salvar e carregar checkpoint."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    trainer.train(train_loader, val_loader, epochs=3)

    checkpoint_path = str(tmp_path / "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)

    # Carregar em um novo trainer
    new_model = LSTMPredictor(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    new_trainer = ModelTrainer(new_model)
    new_trainer.load_checkpoint(checkpoint_path)

    # Verificar que os parametros sao iguais
    for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
        assert torch.allclose(p1, p2)


# --- MLflow Logging Tests ---


def test_train_without_mlflow_installed(trainer, dummy_data):
    """Trainer deve funcionar sem MLflow instalado."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    with patch("src.models.trainer._MLFLOW_AVAILABLE", False):
        history = trainer.train(train_loader, val_loader, epochs=2)

    assert "train_losses" in history
    assert len(history["train_losses"]) == 2


def test_train_without_mlflow_active(trainer, dummy_data):
    """Trainer deve funcionar com MLflow instalado mas sem run ativa."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    with patch("src.models.trainer._MLFLOW_AVAILABLE", True):
        with patch("mlflow.active_run", return_value=None):
            history = trainer.train(train_loader, val_loader, epochs=2)

    assert "train_losses" in history
    assert len(history["train_losses"]) == 2


def test_train_logs_to_mlflow_when_active(trainer, dummy_data):
    """Trainer deve logar métricas quando MLflow está ativo."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    mock_run = MagicMock()

    with patch("src.models.trainer._MLFLOW_AVAILABLE", True):
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("mlflow.log_metric") as mock_log:
                trainer.train(train_loader, val_loader, epochs=2)

    assert mock_log.called
    call_args = [call[0][0] for call in mock_log.call_args_list]
    assert "train_loss" in call_args
    assert "val_loss" in call_args


def test_evaluate_logs_metrics_when_mlflow_active(trainer, dummy_data):
    """evaluate deve logar métricas quando MLflow está ativo."""
    X, y = dummy_data
    test_loader = trainer.create_dataloader(X, y, batch_size=16, shuffle=False)

    mock_run = MagicMock()

    with patch("src.models.trainer._MLFLOW_AVAILABLE", True):
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("mlflow.log_metrics") as mock_log:
                trainer.evaluate(test_loader)

    mock_log.assert_called_once()
    logged_metrics = mock_log.call_args[0][0]
    assert "eval_mae" in logged_metrics
    assert "eval_rmse" in logged_metrics
    assert "eval_mape" in logged_metrics
    assert "eval_r2_score" in logged_metrics


def test_logging_failure_does_not_break_training(trainer, dummy_data):
    """Falha no logging não deve quebrar o treino."""
    X, y = dummy_data
    train_loader = trainer.create_dataloader(X[:40], y[:40], batch_size=16)
    val_loader = trainer.create_dataloader(X[40:], y[40:], batch_size=10, shuffle=False)

    mock_run = MagicMock()

    with patch("src.models.trainer._MLFLOW_AVAILABLE", True):
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("mlflow.log_metric", side_effect=Exception("MLflow error")):
                history = trainer.train(train_loader, val_loader, epochs=2)

    assert len(history["train_losses"]) == 2
