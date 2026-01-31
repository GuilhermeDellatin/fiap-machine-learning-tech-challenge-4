"""
Script CLI para treinamento do modelo.

Uso:
    python scripts/train.py --ticker PETR4.SA --epochs 100
    python scripts/train.py --ticker AAPL --epochs 50 --hidden-size 128
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Adicionar src ao path
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
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")

    parser.add_argument("--ticker", type=str, required=True,
                        help="Stock ticker (e.g., PETR4.SA, AAPL)")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS,
                        help=f"Number of epochs (default: {settings.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE,
                        help=f"Batch size (default: {settings.BATCH_SIZE})")
    parser.add_argument("--sequence-length", type=int, default=settings.SEQUENCE_LENGTH,
                        help=f"Sequence length (default: {settings.SEQUENCE_LENGTH})")
    parser.add_argument("--hidden-size", type=int, default=settings.HIDDEN_SIZE,
                        help=f"LSTM hidden size (default: {settings.HIDDEN_SIZE})")
    parser.add_argument("--num-layers", type=int, default=settings.NUM_LAYERS,
                        help=f"LSTM layers (default: {settings.NUM_LAYERS})")
    parser.add_argument("--dropout", type=float, default=settings.DROPOUT,
                        help=f"Dropout rate (default: {settings.DROPOUT})")
    parser.add_argument("--learning-rate", type=float, default=settings.LEARNING_RATE,
                        help=f"Learning rate (default: {settings.LEARNING_RATE})")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip data sync (use existing cache)")
    parser.add_argument("--no-activate", action="store_true",
                        help="Don't activate model after training")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of historical data (default: 5)")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"Training LSTM for {args.ticker}")
    logger.info("=" * 60)

    # Inicializar banco
    init_db()
    db = SessionLocal()

    try:
        # 1. Coletar dados
        logger.info("Step 1: Collecting data...")
        collector = StockDataCollector()

        if args.no_sync:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")
            df = collector.download_data(db, args.ticker, start_date, end_date)
        else:
            df = collector.sync_data(db, args.ticker, years=args.years)

        logger.info(f"Collected {len(df)} records")

        # 2. Preprocessar
        logger.info("Step 2: Preprocessing...")
        preprocessor = DataPreprocessor()
        scaled_data = preprocessor.fit_transform(df)

        X, y = preprocessor.create_sequences(scaled_data, args.sequence_length)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)

        # 3. Criar modelo
        logger.info("Step 3: Creating model...")
        model = LSTMPredictor(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

        # 4. Treinar
        logger.info("Step 4: Training...")
        trainer = ModelTrainer(
            model=model,
            learning_rate=args.learning_rate,
            early_stopping_patience=settings.EARLY_STOPPING_PATIENCE,
        )

        train_loader = trainer.create_dataloader(X_train, y_train, args.batch_size)
        val_loader = trainer.create_dataloader(X_val, y_val, args.batch_size, shuffle=False)
        test_loader = trainer.create_dataloader(X_test, y_test, args.batch_size, shuffle=False)

        history = trainer.train(train_loader, val_loader, args.epochs)

        # 5. Avaliar
        logger.info("Step 5: Evaluating...")
        metrics = trainer.evaluate(test_loader)

        # 6. Salvar modelo
        logger.info("Step 6: Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{args.ticker}_{timestamp}"

        model_path = f"{settings.MODEL_DIR}/{version_id}.pt"
        scaler_path = f"{settings.MODEL_DIR}/{version_id}_scaler.joblib"

        Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_path)
        preprocessor.save_scaler(scaler_path)

        # 7. Registrar no banco
        logger.info("Step 7: Registering model...")
        hyperparams = {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
        }

        registry = ModelRegistryRepository()
        registered = registry.register_model(
            db=db,
            ticker=args.ticker,
            model_path=model_path,
            scaler_path=scaler_path,
            metrics=metrics,
            hyperparams=hyperparams,
            epochs=history.get("best_epoch", args.epochs),
        )

        # 8. Ativar modelo
        if not args.no_activate:
            logger.info("Step 8: Activating model...")
            registry.set_active_model(db, args.ticker, registered.version_id)

        db.commit()

        # Resultado final
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Version: {registered.version_id}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r2_score']:.4f}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Scaler: {scaler_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
