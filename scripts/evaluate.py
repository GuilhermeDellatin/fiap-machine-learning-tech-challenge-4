"""
Script para avaliar modelo existente.

Uso:
    python scripts/evaluate.py --ticker PETR4.SA
    python scripts/evaluate.py --version-id PETR4.SA_20240115_143022
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import SessionLocal, init_db
from src.database.repository import ModelRegistryRepository
from src.data.collector import StockDataCollector
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str,
                       help="Ticker (uses active model)")
    group.add_argument("--version-id", type=str,
                       help="Specific model version")

    parser.add_argument("--output", type=str, default="evaluation_report.json",
                        help="Output JSON file")

    return parser.parse_args()


def main():
    args = parse_args()

    init_db()
    db = SessionLocal()

    try:
        registry = ModelRegistryRepository()

        # Buscar modelo
        if args.version_id:
            model_info = registry.get_model(db, args.version_id)
        else:
            model_info = registry.get_active_model(db, args.ticker)

        if not model_info:
            logger.error("Model not found")
            return

        logger.info(f"Evaluating: {model_info.version_id}")

        # Carregar modelo
        model = LSTMPredictor(
            hidden_size=settings.HIDDEN_SIZE,
            num_layers=settings.NUM_LAYERS,
        )

        trainer = ModelTrainer(model, learning_rate=0.001)
        trainer.load_checkpoint(model_info.model_path)

        # Carregar dados
        collector = StockDataCollector()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = collector.download_data(db, model_info.ticker, start_date, end_date)

        # Preprocessar
        preprocessor = DataPreprocessor()
        preprocessor.load_scaler(model_info.scaler_path)
        scaled = preprocessor.transform(df)
        X, y = preprocessor.create_sequences(scaled, settings.SEQUENCE_LENGTH)

        # Avaliar
        test_loader = trainer.create_dataloader(X, y, batch_size=32, shuffle=False)
        metrics = trainer.evaluate(test_loader)

        # Salvar relatório
        report = {
            "version_id": model_info.version_id,
            "ticker": model_info.ticker,
            "evaluated_at": datetime.now().isoformat(),
            "metrics": metrics,
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {args.output}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r2_score']:.4f}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
