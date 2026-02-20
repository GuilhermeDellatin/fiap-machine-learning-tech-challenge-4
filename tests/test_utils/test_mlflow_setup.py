"""
Testes para utilitários MLflow.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestGenerateVersionId:
    """Testes para generate_version_id."""

    def test_format_correct(self):
        """version_id deve ter formato {ticker}_{timestamp}."""
        from src.utils.mlflow_setup import generate_version_id

        version_id = generate_version_id("AAPL")

        # Deve começar com ticker
        assert version_id.startswith("AAPL_")

        # Deve ter formato de timestamp
        parts = version_id.split("_")
        assert len(parts) == 3  # AAPL_YYYYMMDD_HHMMSS
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS

    def test_unique_ids(self):
        """IDs gerados devem ser únicos."""
        from src.utils.mlflow_setup import generate_version_id
        import time

        id1 = generate_version_id("AAPL")
        time.sleep(1)  # Garantir timestamp diferente
        id2 = generate_version_id("AAPL")

        assert id1 != id2

    def test_preserves_ticker_case(self):
        """Ticker deve ser usado como fornecido."""
        from src.utils.mlflow_setup import generate_version_id

        version_id = generate_version_id("petr4.sa")
        assert version_id.startswith("petr4.sa_")


class TestSetupMlflow:
    """Testes para setup_mlflow."""

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.create_experiment', return_value="123")
    @patch('mlflow.get_experiment_by_name', return_value=None)
    def test_creates_experiment_if_not_exists(
        self, mock_get, mock_create, mock_set_exp, mock_set_uri
    ):
        """Deve criar experimento se não existir."""
        from src.utils.mlflow_setup import setup_mlflow
        import src.utils.mlflow_setup as module

        # Reset flag para testar do zero
        module._mlflow_configured = False

        experiment_id = setup_mlflow(
            tracking_uri="sqlite:///test.db",
            experiment_name="test-experiment"
        )

        mock_create.assert_called_once()
        assert experiment_id == "123"

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.create_experiment')
    @patch('mlflow.get_experiment_by_name')
    def test_uses_existing_experiment(
        self, mock_get, mock_create, mock_set_exp, mock_set_uri
    ):
        """Deve usar experimento existente se já existir."""
        from src.utils.mlflow_setup import setup_mlflow
        import src.utils.mlflow_setup as module

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "456"
        mock_get.return_value = mock_experiment

        module._mlflow_configured = False

        experiment_id = setup_mlflow()

        mock_create.assert_not_called()
        assert experiment_id == "456"

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.get_experiment_by_name')
    def test_idempotent_skips_reconfiguration(
        self, mock_get, mock_set_exp, mock_set_uri
    ):
        """Chamadas repetidas não devem reconfigurar."""
        from src.utils.mlflow_setup import setup_mlflow
        import src.utils.mlflow_setup as module

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "789"
        mock_get.return_value = mock_experiment

        # Simular que já foi configurado
        module._mlflow_configured = True

        experiment_id = setup_mlflow()

        # set_tracking_uri NÃO deve ser chamado (já configurado)
        mock_set_uri.assert_not_called()
        assert experiment_id == "789"

        # Resetar para não afetar outros testes
        module._mlflow_configured = False


class TestIsMlflowActive:
    """Testes para is_mlflow_active."""

    @patch('mlflow.active_run', return_value=None)
    def test_returns_false_when_no_run(self, mock_active):
        """Deve retornar False quando não há run ativa."""
        from src.utils.mlflow_setup import is_mlflow_active
        assert is_mlflow_active() is False

    @patch('mlflow.active_run', return_value=MagicMock())
    def test_returns_true_when_run_active(self, mock_active):
        """Deve retornar True quando há run ativa."""
        from src.utils.mlflow_setup import is_mlflow_active
        assert is_mlflow_active() is True
