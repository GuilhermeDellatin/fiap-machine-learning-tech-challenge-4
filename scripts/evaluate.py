"""
Script to evaluate an existing trained model.

Usage:
    python scripts/evaluate.py --ticker PETR4.SA
    python scripts/evaluate.py --version-id PETR4.SA_20240115_143022
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import StockDataCollector
from src.data.preprocessor import DataPreprocessor
from src.database.connection import SessionLocal, init_db
from src.database.repository import ModelRegistryRepository
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer
from src.monitoring.mlflow_tracker import MLflowTracker
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Ticker (uses active model)")
    group.add_argument("--version-id", type=str, help="Specific model version")

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output JSON report path",
    )

    return parser.parse_args()


def _load_hyperparameters(raw_hyperparams: str | None) -> dict:
    if not raw_hyperparams:
        return {}

    try:
        loaded = json.loads(raw_hyperparams)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        logger.warning("Could not parse stored hyperparameters. Falling back to defaults.")

    return {}


def main():
    args = parse_args()

    init_db()
    db = SessionLocal()
    tracker = MLflowTracker(
        enabled=settings.MLFLOW_ENABLED,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
    )

    try:
        registry = ModelRegistryRepository()

        if args.version_id:
            model_info = registry.get_model(db, args.version_id)
        else:
            model_info = registry.get_active_model(db, args.ticker)

        if not model_info:
            logger.error("Model not found")
            return

        logger.info(f"Evaluating: {model_info.version_id}")

        hyperparams = _load_hyperparameters(model_info.hyperparameters)
        hidden_size = int(hyperparams.get("hidden_size", settings.HIDDEN_SIZE))
        num_layers = int(hyperparams.get("num_layers", settings.NUM_LAYERS))
        dropout = float(hyperparams.get("dropout", settings.DROPOUT))
        sequence_length = int(hyperparams.get("sequence_length", settings.SEQUENCE_LENGTH))
        batch_size = int(hyperparams.get("batch_size", settings.BATCH_SIZE))

        run_name = (
            f"cli_evaluate_{model_info.version_id}_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        run_tags = {
            "pipeline": "cli",
            "stage": "evaluation",
            "ticker": model_info.ticker,
            "model_version_id": model_info.version_id,
        }

        with tracker.start_run(run_name=run_name, tags=run_tags):
            tracker.log_params(
                {
                    "ticker": model_info.ticker,
                    "model_version_id": model_info.version_id,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                }
            )

            model = LSTMPredictor(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

            trainer = ModelTrainer(model, learning_rate=0.001)
            trainer.load_checkpoint(model_info.model_path)

            collector = StockDataCollector()
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            df = collector.download_data(db, model_info.ticker, start_date, end_date)

            preprocessor = DataPreprocessor()
            preprocessor.load_scaler(model_info.scaler_path)
            scaled = preprocessor.transform(df)
            X, y = preprocessor.create_sequences(scaled, sequence_length)

            test_loader = trainer.create_dataloader(X, y, batch_size=batch_size, shuffle=False)
            metrics = trainer.evaluate(test_loader)
            tracker.log_metrics(metrics)
            tracker.log_params({"evaluation_samples": len(X)})

            report = {
                "version_id": model_info.version_id,
                "ticker": model_info.ticker,
                "evaluated_at": datetime.now().isoformat(),
                "metrics": metrics,
                "hyperparameters": {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                },
            }

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            tracker.log_artifact(args.output, artifact_path="reports")

            logger.info(f"Report saved to {args.output}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            logger.info(f"R2: {metrics['r2_score']:.4f}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
