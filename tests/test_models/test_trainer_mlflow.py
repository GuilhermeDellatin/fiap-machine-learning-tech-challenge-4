"""
Testes de integração Trainer + MLflow.
"""
import pytest
from unittest.mock import patch, MagicMock, call
import torch
import numpy as np


@pytest.fixture
def simple_model():
    """Modelo LSTM simples para testes."""
    from src.models.lstm_model import LSTMPredictor
    return LSTMPredictor(input_size=1, hidden_size=16, num_layers=1)


@pytest.fixture
def sample_loaders(simple_model):
    """DataLoaders com dados sintéticos."""
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer(simple_model, learning_rate=0.01)

    # Dados sintéticos
    X = np.random.rand(100, 10, 1).astype(np.float32)
    y = np.random.rand(100, 1).astype(np.float32)

    loader = trainer.create_dataloader(X, y, batch_size=16)
    return loader, loader  # train e val iguais para simplificar


class TestTrainerWithMlflowInstalled:
    """Testes do Trainer com MLflow instalado."""

    def test_logs_metrics_when_run_active(self, simple_model, sample_loaders):
        """Deve logar métricas quando há run ativa."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        train_loader, val_loader = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', True), \
             patch('mlflow.active_run', return_value=MagicMock()), \
             patch('mlflow.log_metric') as mock_log_metric:
            trainer.train(train_loader, val_loader, epochs=2)

        # Verificar que métricas foram logadas
        metric_names = [c[0][0] for c in mock_log_metric.call_args_list]
        assert 'train_loss' in metric_names
        assert 'val_loss' in metric_names

    def test_does_not_log_when_no_run_active(self, simple_model, sample_loaders):
        """Não deve logar métricas quando não há run ativa."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        train_loader, val_loader = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', True), \
             patch('mlflow.active_run', return_value=None), \
             patch('mlflow.log_metric') as mock_log_metric:
            trainer.train(train_loader, val_loader, epochs=2)

        mock_log_metric.assert_not_called()

    def test_training_continues_if_logging_fails(self, simple_model, sample_loaders):
        """Treino deve continuar mesmo se logging falhar."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        train_loader, val_loader = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', True), \
             patch('mlflow.active_run', return_value=MagicMock()), \
             patch('mlflow.log_metric', side_effect=Exception("MLflow error")):
            history = trainer.train(train_loader, val_loader, epochs=2)

        assert len(history['train_losses']) == 2

    def test_evaluate_logs_eval_metrics(self, simple_model, sample_loaders):
        """evaluate() deve logar métricas de avaliação com prefixo eval_."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        test_loader, _ = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', True), \
             patch('mlflow.active_run', return_value=MagicMock()), \
             patch('mlflow.log_metrics') as mock_log_metrics:
            trainer.evaluate(test_loader)

        mock_log_metrics.assert_called_once()
        logged = mock_log_metrics.call_args[0][0]
        assert 'eval_mae' in logged
        assert 'eval_rmse' in logged
        assert 'eval_mape' in logged
        assert 'eval_r2_score' in logged


class TestTrainerWithoutMlflow:
    """Testes do Trainer quando MLflow não está instalado."""

    def test_train_works_without_mlflow_installed(self, simple_model, sample_loaders):
        """Trainer deve funcionar normalmente sem MLflow instalado."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        train_loader, val_loader = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', False):
            history = trainer.train(train_loader, val_loader, epochs=2)

        assert "train_losses" in history
        assert len(history["train_losses"]) == 2

    def test_evaluate_works_without_mlflow_installed(self, simple_model, sample_loaders):
        """evaluate() deve funcionar normalmente sem MLflow instalado."""
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(simple_model, learning_rate=0.01)
        test_loader, _ = sample_loaders

        with patch('src.models.trainer._MLFLOW_AVAILABLE', False):
            metrics = trainer.evaluate(test_loader)

        assert 'mae' in metrics
        assert 'rmse' in metrics
