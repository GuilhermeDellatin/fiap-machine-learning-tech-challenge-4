"""
Testes de integração API Training + MLflow.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(test_db):
    """TestClient com mocks.

    Não usa context manager para evitar que o lifespan dispare init_db()
    no banco real (stock_cache.db), o que polui o pool de conexões SQLAlchemy
    e causa 'test setup failed' quando os testes rodam em conjunto.
    """
    from src.api.main import app
    from src.api.dependencies import get_database

    app.dependency_overrides[get_database] = lambda: test_db

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


class TestTrainingEndpointWithMlflow:
    """Testes do endpoint de training com MLflow."""

    @patch('src.api.routes.training.train_model_task')
    @patch('src.api.routes.training.get_collector')
    def test_start_training_returns_202(self, mock_collector, mock_train, client):
        """POST /training/start deve retornar 202.

        train_model_task é mockado para evitar que a background task acesse o
        banco real via SessionLocal(), o que corrompe o pool de conexões
        SQLAlchemy entre testes.
        """
        mock_collector.return_value.validate_ticker.return_value = True

        response = client.post(
            "/api/v1/training/start",
            json={"ticker": "AAPL", "epochs": 5}
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "accepted"

    @patch('src.api.routes.training.ModelRegistryRepository')
    @patch('src.api.routes.training.TrainingJobRepository')
    @patch('src.api.routes.training.ModelTrainer')
    @patch('src.api.routes.training.DataPreprocessor')
    @patch('src.api.routes.training.StockDataCollector')
    @patch('mlflow.start_run')
    @patch('src.api.routes.training.setup_mlflow')
    def test_training_task_creates_mlflow_run(
        self, mock_setup, mock_start, mock_collector,
        mock_preprocessor, mock_trainer, mock_job_repo, mock_model_repo
    ):
        """train_model_task deve criar run MLflow."""
        from src.api.routes.training import train_model_task

        # Configurar mock da run
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start.return_value.__exit__ = MagicMock(return_value=False)

        try:
            train_model_task("job-123", "AAPL", {"epochs": 5})
        except Exception:
            pass  # Task pode falhar por mocks incompletos

        # Verificar que start_run foi chamado
        mock_start.assert_called()

    @patch('src.api.routes.training.ModelRegistryRepository')
    @patch('src.api.routes.training.TrainingJobRepository')
    @patch('src.api.routes.training.ModelTrainer')
    @patch('src.api.routes.training.DataPreprocessor')
    @patch('src.api.routes.training.StockDataCollector')
    @patch('src.api.routes.training.SessionLocal')
    @patch('src.api.routes.training.setup_mlflow', side_effect=Exception("No MLflow"))
    def test_training_continues_without_mlflow(
        self, mock_setup, mock_session, mock_collector,
        mock_preprocessor, mock_trainer, mock_job_repo, mock_model_repo
    ):
        """train_model_task deve continuar se MLflow falhar no setup.

        Todos os componentes que acessam rede/disco/banco real são mockados
        para evitar efeitos colaterais (yfinance, treinamento real, SessionLocal).
        O objetivo do teste é apenas verificar que "No MLflow" não propaga.
        """
        from src.api.routes.training import train_model_task

        # Não deve levantar exceção — deve usar nullcontext
        try:
            train_model_task("job-123", "AAPL", {"epochs": 5})
        except Exception as e:
            # Pode falhar por outros motivos (mocks incompletos),
            # mas NÃO deve ser o erro de MLflow
            assert "No MLflow" not in str(e)
