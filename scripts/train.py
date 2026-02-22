"""
Script CLI para treinamento do modelo com tracking MLflow.


Uso:
   python scripts/train.py --ticker PETR4.SA --epochs 100
   python scripts/train.py --ticker AAPL --epochs 50 --hidden-size 128
   python scripts/train.py --ticker AAPL --epochs 50 --no-mlflow


O MLflow UI mostra os experimentos em:
   http://localhost:5000
"""
import argparse
import sys
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch

from src.database.connection import SessionLocal, init_db
from src.database.repository import ModelRegistryRepository
from src.data.collector import StockDataCollector
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.mlflow_setup import setup_mlflow, generate_version_id
from src.utils.mlflow_tracing import tracing

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model with MLflow tracking")

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
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")

    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    logger.info("=" * 60)
    logger.info(f"Training LSTM for {ticker}")
    logger.info("=" * 60)

    # =========================================================
    # 1. SETUP INICIAL
    # =========================================================
    init_db()
    db = SessionLocal()

    # =========================================================
    # 2. CONFIGURAR MLFLOW (se habilitado)
    # =========================================================
    mlflow_enabled = not args.no_mlflow
    if mlflow_enabled:
        setup_mlflow()
        logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")

    # =========================================================
    # 3. GERAR VERSION_ID ANTES DO TREINO
    # =========================================================
    version_id = generate_version_id(ticker)
    logger.info(f"Version ID: {version_id}")

    # Definir paths com version_id
    model_path = f"{settings.MODEL_DIR}/{version_id}.pt"
    scaler_path = f"{settings.MODEL_DIR}/{version_id}_scaler.joblib"

    # Garantir que diretório existe
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 4. PREPARAR HIPERPARÂMETROS
    # =========================================================
    hyperparams = {
        "ticker": ticker,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": settings.EARLY_STOPPING_PATIENCE,
        "years_of_data": args.years,
    }

    try:
        # =========================================================
        # 5. CONTEXTO MLFLOW (Orchestrator abre a run)
        # =========================================================
        # Se MLflow desabilitado, usar nullcontext
        if mlflow_enabled:
            run_context = mlflow.start_run(run_name=version_id)
        else:
            run_context = nullcontext()

        with run_context as run:
            # Logar informações da run
            if mlflow_enabled and run:
                run_id = run.info.run_id
                logger.info(f"MLflow Run ID: {run_id}")

                # Logar parâmetros
                mlflow.log_params(hyperparams)

                # Tags para organização
                mlflow.set_tags({
                    "source": "cli",
                    "ticker": ticker,
                    "version_id": version_id,
                })
            else:
                run_id = None

            # =========================================================
            # TRACE: PIPELINE DE TREINAMENTO CLI
            # =========================================================
            with tracing.pipeline("training_pipeline_cli",
                                  inputs={"ticker": ticker, "hyperparams": hyperparams, "source": "cli"}) as root_span:

                # =========================================================
                # 6. COLETAR DADOS
                # =========================================================
                logger.info("Step 1: Collecting data...")
                with tracing.data_collection(ticker, no_sync=args.no_sync, years=args.years) as span:
                    collector = StockDataCollector()

                    if args.no_sync:
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")
                        df = collector.download_data(db, ticker, start_date, end_date)
                    else:
                        df = collector.sync_data(db, ticker, years=args.years)

                    logger.info(f"Collected {len(df)} records")
                    span.set_outputs({"records": len(df)})
                    span.set_attributes({"dataset.rows": len(df), "dataset.columns": len(df.columns)})

                # Logar tamanho do dataset
                if mlflow_enabled:
                    mlflow.log_metric("dataset_size", len(df))

                # =========================================================
                # 7. PREPROCESSAR
                # =========================================================
                logger.info("Step 2: Preprocessing...")
                with tracing.preprocessing(args.sequence_length, len(df)) as span:
                    preprocessor = DataPreprocessor()
                    scaled_data = preprocessor.fit_transform(df)

                    X, y = preprocessor.create_sequences(scaled_data, args.sequence_length)
                    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)
                    span.set_outputs({
                        "train_size": len(X_train),
                        "val_size": len(X_val),
                        "test_size": len(X_test),
                        "total_sequences": len(X),
                    })

                # Logar tamanhos dos splits
                if mlflow_enabled:
                    mlflow.log_metrics({
                        "train_size": len(X_train),
                        "val_size": len(X_val),
                        "test_size": len(X_test),
                    })

                # =========================================================
                # 8. CRIAR MODELO
                # =========================================================
                logger.info("Step 3: Creating model...")
                with tracing.model_creation(
                        input_size=1,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                ) as span:
                    model = LSTMPredictor(
                        input_size=1,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                    )
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    span.set_outputs({"total_parameters": total_params, "trainable_parameters": trainable_params})

                # =========================================================
                # 9. TREINAR (Trainer loga métricas passivamente)
                # =========================================================
                logger.info("Step 4: Training...")
                with tracing.model_training(args.epochs, args.learning_rate, args.batch_size) as span:
                    trainer = ModelTrainer(
                        model=model,
                        learning_rate=args.learning_rate,
                        early_stopping_patience=settings.EARLY_STOPPING_PATIENCE,
                    )

                    train_loader = trainer.create_dataloader(X_train, y_train, args.batch_size)
                    val_loader = trainer.create_dataloader(X_val, y_val, args.batch_size, shuffle=False)
                    test_loader = trainer.create_dataloader(X_test, y_test, args.batch_size, shuffle=False)

                    history = trainer.train(train_loader, val_loader, args.epochs)
                    span.set_outputs({
                        "best_epoch": history["best_epoch"],
                        "total_epochs_run": len(history["train_losses"]),
                        "final_train_loss": history["train_losses"][-1],
                        "final_val_loss": history["val_losses"][-1],
                    })

                # =========================================================
                # 10. AVALIAR (Trainer loga métricas passivamente)
                # =========================================================
                logger.info("Step 5: Evaluating...")
                with tracing.model_evaluation(len(X_test)) as span:
                    metrics = trainer.evaluate(test_loader)
                    span.set_outputs(metrics)

                # =========================================================
                # 11. SALVAR MODELO
                # =========================================================
                logger.info("Step 6: Saving model...")
                with tracing.save_model(model_path, scaler_path) as span:
                    trainer.save_checkpoint(model_path)
                    preprocessor.save_scaler(scaler_path)
                    span.set_outputs({"model_saved": True})

                # =========================================================
                # 12. LOGAR ARTIFACTS NO MLFLOW
                # =========================================================
                if mlflow_enabled:
                    logger.info("Step 7: Logging artifacts to MLflow...")
                    with tracing.artifact_logging(model_path, scaler_path) as span:
                        mlflow.log_artifact(model_path, artifact_path="model")
                        mlflow.log_artifact(scaler_path, artifact_path="model")

                        model_cpu = model.cpu()
                        mlflow.pytorch.log_model(
                            model_cpu,
                            artifact_path="lstm_model",
                            registered_model_name=None,
                        )

                        mlflow.set_tag("status", "completed")
                        span.set_outputs({"artifacts_logged": True})

                # =========================================================
                # 13. REGISTRAR NO MODELREGISTRY (SQLite)
                # =========================================================
                logger.info("Step 8: Registering model...")
                with tracing.model_registration(ticker, version_id) as span:
                    registry = ModelRegistryRepository()
                    registry.register_model(
                        db=db,
                        ticker=ticker,
                        version_id=version_id,
                        model_path=model_path,
                        scaler_path=scaler_path,
                        metrics=metrics,
                        hyperparams=hyperparams,
                        epochs=history.get("best_epoch", args.epochs),
                        mlflow_run_id=run_id,
                    )

                    # =========================================================
                    # 14. ATIVAR MODELO
                    # =========================================================
                    if not args.no_activate:
                        logger.info("Step 9: Activating model...")
                        registry.set_active_model(db, ticker, version_id)

                    span.set_outputs({"registered": True, "activated": not args.no_activate})

                db.commit()

                root_span.set_outputs({
                    "version_id": version_id,
                    "run_id": run_id,
                    "metrics": metrics,
                    "status": "completed",
                })

        # =========================================================
        # 15. RESULTADO FINAL
        # =========================================================
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Version: {version_id}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r2_score']:.4f}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Scaler: {scaler_path}")
        if mlflow_enabled:
            logger.info(f"MLflow Run: {run_id}")
            logger.info(f"MLflow UI: http://localhost:5000")
        logger.info("=" * 60)


    except Exception as e:
        logger.error(f"Training failed: {e}")
        db.rollback()

        # Logar falha no MLflow
        if mlflow_enabled and mlflow.active_run():
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e)[:250])  # Limitar tamanho

        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
