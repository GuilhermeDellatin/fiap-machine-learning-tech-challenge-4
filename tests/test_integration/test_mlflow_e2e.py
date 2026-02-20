"""
Teste de integração end-to-end com MLflow real (não mockado).

Roda contra um tracking URI em memória para validar compatibilidade
de versão e serialização real.

Requer: mlflow>=2.14.0 instalado
Rodar: pytest tests/test_integration/test_mlflow_e2e.py -v -m integration
"""
import pytest
import tempfile
import os

import mlflow
import numpy as np

from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer
from src.utils.mlflow_setup import setup_mlflow, generate_version_id


@pytest.mark.integration
class TestMlflowEndToEnd:
    """Teste E2E com MLflow real."""

    def test_full_training_cycle_with_real_mlflow(self):
        """Ciclo completo: setup → run → train → log → close."""
        # ignore_cleanup_errors=True evita falha de limpeza no Windows,
        # onde o SQLite pode manter o arquivo bloqueado após uso.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/test_mlflow.db"

            # 1. Setup
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("test-experiment")

            # 2. Gerar version_id
            version_id = generate_version_id("TEST")

            # 3. Criar modelo e dados sintéticos
            model = LSTMPredictor(input_size=1, hidden_size=16, num_layers=1)
            trainer = ModelTrainer(model, learning_rate=0.01)

            X = np.random.rand(50, 10, 1).astype(np.float32)
            y = np.random.rand(50, 1).astype(np.float32)
            loader = trainer.create_dataloader(X, y, batch_size=16)

            # 4. Treinar dentro de run
            with mlflow.start_run(run_name=version_id) as run:
                run_id = run.info.run_id

                mlflow.log_params({"hidden_size": 16, "lr": 0.01})
                mlflow.set_tag("version_id", version_id)

                # Trainer deve logar métricas passivamente
                history = trainer.train(loader, loader, epochs=3)
                metrics = trainer.evaluate(loader)

                mlflow.log_metrics({
                    "final_mae": metrics["mae"],
                    "final_rmse": metrics["rmse"]
                })

            # 5. Verificar que run foi registrada
            run_data = mlflow.get_run(run_id)

            assert run_data.info.status == "FINISHED"
            assert run_data.data.tags["version_id"] == version_id
            assert "hidden_size" in run_data.data.params
            assert "final_mae" in run_data.data.metrics

            # Verificar que métricas do Trainer passivo foram logadas
            assert "train_loss" in run_data.data.metrics
            assert "val_loss" in run_data.data.metrics
            assert "eval_mae" in run_data.data.metrics

            # Liberar conexões antes do cleanup do diretório temporário
            mlflow.set_tracking_uri("")
