"""
Endpoints de treinamento com integração MLflow.
"""
import json
import uuid
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.pytorch
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from src.api.dependencies import get_database, get_collector
from src.database.connection import SessionLocal
from src.database.repository import (
    TrainingJobRepository,
    ModelRegistryRepository,
    PriceCacheRepository,
)
from src.data.collector import StockDataCollector
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import ModelTrainer
from src.api.schemas.training import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
    TrainingJobListResponse,
    TrainingJobSummary,
    ModelListResponse,
    ModelInfo,
    ActivateModelResponse,
)
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.mlflow_setup import setup_mlflow, generate_version_id
from src.utils.mlflow_tracing import tracing

router = APIRouter()
logger = get_logger(__name__)


def train_model_task(job_id: str, ticker: str, params: dict) -> None:
    """
    Background task para treinamento com MLflow tracking.


    Esta função é o ORCHESTRATOR para treinos via API.
    Ela é responsável por:
    1. Configurar MLflow (com fallback se falhar)
    2. Gerar version_id consistente
    3. Gerenciar o ciclo de vida da run
    4. Registrar no ModelRegistry com o mesmo version_id


    Args:
        job_id: UUID do job de treinamento
        ticker: Código da ação
        params: Dict com hiperparâmetros
    """
    db = SessionLocal()
    job_repo = TrainingJobRepository()
    model_repo = ModelRegistryRepository()

    # =========================================================
    # 1. CONFIGURAR MLFLOW (com fallback)
    # =========================================================
    mlflow_enabled = True
    try:
        setup_mlflow()
    except Exception as e:
        logger.warning(f"Falha ao configurar MLflow: {e}. Continuando sem tracking.")
        mlflow_enabled = False

    # =========================================================
    # 2. GERAR VERSION_ID ANTES DO TREINO
    # =========================================================
    version_id = generate_version_id(ticker)

    # Definir paths
    model_path = f"{settings.MODEL_DIR}/{version_id}.pt"
    scaler_path = f"{settings.MODEL_DIR}/{version_id}_scaler.joblib"

    # Garantir diretório
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 3. PREPARAR HIPERPARÂMETROS COMPLETOS
    # =========================================================
    hyperparams = {
        "ticker": ticker,
        "job_id": job_id,
        "hidden_size": params.get("hidden_size", settings.HIDDEN_SIZE),
        "num_layers": params.get("num_layers", settings.NUM_LAYERS),
        "dropout": params.get("dropout", settings.DROPOUT),
        "learning_rate": params.get("learning_rate", settings.LEARNING_RATE),
        "sequence_length": params.get("sequence_length", settings.SEQUENCE_LENGTH),
        "batch_size": params.get("batch_size", settings.BATCH_SIZE),
        "epochs": params.get("epochs", settings.EPOCHS),
    }

    run_id = None

    try:
        # =========================================================
        # 4. ABRIR RUN MLFLOW (com fallback nullcontext)
        # =========================================================
        if mlflow_enabled:
            run_context = mlflow.start_run(run_name=version_id)
        else:
            run_context = nullcontext()

        with run_context as run:
            if mlflow_enabled and run:
                run_id = run.info.run_id

                # Logar parâmetros
                mlflow.log_params(hyperparams)

                # Tags para organização e rastreabilidade
                mlflow.set_tags({
                    "source": "api",
                    "job_id": job_id,
                    "ticker": ticker,
                    "version_id": version_id,
                })

                logger.info(f"MLflow Run iniciada: {run_id} (version_id={version_id})")

                # =========================================================
                # TRACE: PIPELINE DE TREINAMENTO
                # =========================================================
                with tracing.pipeline("training_pipeline", inputs={"ticker": ticker, "job_id": job_id,
                                                                   "hyperparams": hyperparams}) as root_span:

                    # =========================================================
                    # 5. ATUALIZAR JOB STATUS: RUNNING
                    # =========================================================
                    job_repo.update_job(
                        db, job_id,
                        status="running",
                        started_at=datetime.utcnow(),
                    )
                    db.commit()

                    # =========================================================
                    # 6. COLETAR DADOS
                    # =========================================================
                    with tracing.data_collection(ticker) as span:
                        collector = StockDataCollector()
                        df = collector.sync_data(db, ticker)
                        span.set_outputs({"records": len(df)})
                        span.set_attributes({"dataset.rows": len(df), "dataset.columns": len(df.columns)})

                    if mlflow_enabled:
                        mlflow.log_metric("dataset_size", len(df))

                    # =========================================================
                    # 7. PREPROCESSAR
                    # =========================================================
                    with tracing.preprocessing(hyperparams["sequence_length"], len(df)) as span:
                        preprocessor = DataPreprocessor()
                        scaled = preprocessor.fit_transform(df)
                        X, y = preprocessor.create_sequences(scaled, hyperparams["sequence_length"])
                        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)
                        span.set_outputs({
                            "train_size": len(X_train),
                            "val_size": len(X_val),
                            "test_size": len(X_test),
                            "total_sequences": len(X),
                        })

                    if mlflow_enabled:
                        mlflow.log_metrics({
                            "train_size": len(X_train),
                            "val_size": len(X_val),
                            "test_size": len(X_test),
                        })

                    # =========================================================
                    # 8. CRIAR MODELO
                    # =========================================================
                    with tracing.model_creation(
                            input_size=1,
                            hidden_size=hyperparams["hidden_size"],
                            num_layers=hyperparams["num_layers"],
                            dropout=hyperparams["dropout"],
                    ) as span:
                        model = LSTMPredictor(
                            input_size=1,
                            hidden_size=hyperparams["hidden_size"],
                            num_layers=hyperparams["num_layers"],
                            dropout=hyperparams["dropout"],
                        )
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        span.set_outputs({"total_parameters": total_params, "trainable_parameters": trainable_params})

                    # =========================================================
                    # 9. TREINAR (Trainer loga métricas passivamente)
                    # =========================================================
                    with tracing.model_training(
                            hyperparams["epochs"], hyperparams["learning_rate"], hyperparams["batch_size"]
                    ) as span:
                        trainer = ModelTrainer(model, learning_rate=hyperparams["learning_rate"])

                        train_loader = trainer.create_dataloader(
                            X_train, y_train, hyperparams["batch_size"]
                        )
                        val_loader = trainer.create_dataloader(
                            X_val, y_val, hyperparams["batch_size"], shuffle=False
                        )
                        test_loader = trainer.create_dataloader(
                            X_test, y_test, hyperparams["batch_size"], shuffle=False
                        )

                        def progress_callback(epoch, total_epochs, train_loss, val_loss, best_loss):
                            job_repo.update_job(
                                db, job_id,
                                epochs_completed=epoch,
                                current_loss=val_loss,
                                best_loss=best_loss,
                            )
                            db.commit()

                        history = trainer.train(
                            train_loader, val_loader, hyperparams["epochs"], progress_callback
                        )
                        span.set_outputs({
                            "best_epoch": history["best_epoch"],
                            "total_epochs_run": len(history["train_losses"]),
                            "final_train_loss": history["train_losses"][-1],
                            "final_val_loss": history["val_losses"][-1],
                        })

                    # =========================================================
                    # 10. AVALIAR
                    # =========================================================
                    with tracing.model_evaluation(len(X_test)) as span:
                        metrics = trainer.evaluate(test_loader)
                        span.set_outputs(metrics)

                    # =========================================================
                    # 11. SALVAR MODELO
                    # =========================================================
                    with tracing.save_model(model_path, scaler_path) as span:
                        trainer.save_checkpoint(model_path)
                        preprocessor.save_scaler(scaler_path)
                        span.set_outputs({"model_saved": True})

                    # =========================================================
                    # 12. LOGAR ARTIFACTS NO MLFLOW
                    # =========================================================
                    if mlflow_enabled:
                        with tracing.artifact_logging(model_path, scaler_path) as span:
                            try:
                                mlflow.log_artifact(model_path, artifact_path="model")
                                mlflow.log_artifact(scaler_path, artifact_path="model")

                                model_cpu = model.cpu()
                                mlflow.pytorch.log_model(
                                    model_cpu,
                                    artifact_path="lstm_model",
                                    registered_model_name=None,
                                )
                                span.set_outputs({"artifacts_logged": True})
                            except Exception as artifact_err:
                                logger.warning(f"Falha ao logar artifacts no MLflow (não crítico): {artifact_err}")
                                span.set_outputs({"artifacts_logged": False, "error": str(artifact_err)})

                            mlflow.set_tag("status", "completed")

                    # =========================================================
                    # 13. REGISTRAR NO MODELREGISTRY (SQLite)
                    # =========================================================
                    with tracing.model_registration(ticker, version_id) as span:
                        model_repo.register_model(
                            db=db,
                            ticker=ticker,
                            version_id=version_id,
                            model_path=model_path,
                            scaler_path=scaler_path,
                            metrics=metrics,
                            hyperparams=hyperparams,
                            epochs=history.get("best_epoch", hyperparams["epochs"]),
                            mlflow_run_id=run_id,
                        )

                        # =========================================================
                        # 14. ATIVAR MODELO
                        # =========================================================
                        model_repo.set_active_model(db, ticker, version_id)
                        span.set_outputs({"registered": True, "activated": True})

                    # =========================================================
                    # 15. ATUALIZAR JOB STATUS: COMPLETED
                    # =========================================================
                    job_repo.update_job(
                        db, job_id,
                        status="completed",
                        completed_at=datetime.utcnow(),
                        model_version_id=version_id,
                    )
                    db.commit()

                    root_span.set_outputs({
                        "version_id": version_id,
                        "run_id": run_id,
                        "metrics": metrics,
                        "status": "completed",
                    })

            logger.info(f"Training completed: {version_id} (run_id={run_id})")


    except Exception as e:
        logger.error(f"Training failed: {e}")

        # =========================================================
        # LOGAR FALHA NO MLFLOW (espelho do CLI)
        # =========================================================
        if mlflow_enabled and mlflow.active_run():
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e)[:250])

        # Atualizar job como failed
        job_repo.update_job(
            db, job_id,
            status="failed",
            error_message=str(e)[:500],
            completed_at=datetime.utcnow(),
        )
        db.commit()


    finally:
        db.close()


@router.post("/start", status_code=202, response_model=TrainingResponse)
def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_database),
):
    """
    Inicia treinamento em background com MLflow tracking.


    O job é rastreado em dois lugares:
    - TrainingJob (SQLite): Status, progresso, erros
    - MLflow: Métricas, parâmetros, artifacts


    Returns:
        202 Accepted com job_id
    """
    # Validar ticker
    collector = get_collector()
    if not collector.validate_ticker(request.ticker):
        raise HTTPException(404, f"Ticker {request.ticker} not found in yfinance")

    # Criar job
    job_id = str(uuid.uuid4())
    job_repo = TrainingJobRepository()
    job_repo.create_job(
        db, job_id, request.ticker,
        request.epochs, request.model_dump(),
    )
    db.commit()

    # Disparar background task
    background_tasks.add_task(
        train_model_task,
        job_id=job_id,
        ticker=request.ticker,
        params=request.model_dump(),
    )

    return TrainingResponse(
        job_id=job_id,
        status="accepted",
        message=f"Training job started for {request.ticker}. Track in MLflow UI.",
        ticker=request.ticker,
    )


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
def get_training_status(job_id: str, db: Session = Depends(get_database)):
    """Status detalhado do job."""
    repo = TrainingJobRepository()
    job = repo.get_job(db, job_id)

    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    progress = (job.epochs_completed / job.epochs_total * 100) if job.epochs_total > 0 else 0

    return TrainingStatusResponse(
        job_id=job.job_id,
        ticker=job.ticker,
        status=job.status,
        progress_percent=progress,
        epochs_completed=job.epochs_completed,
        epochs_total=job.epochs_total,
        current_loss=job.current_loss,
        best_loss=job.best_loss,
        error_message=job.error_message,
        model_version_id=job.model_version_id,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs", response_model=TrainingJobListResponse)
def list_training_jobs(
        ticker: str = None,
        status: str = None,
        limit: int = 50,
        db: Session = Depends(get_database),
):
    """Lista jobs de treinamento."""
    repo = TrainingJobRepository()
    jobs = repo.list_jobs(db, ticker=ticker, status=status, limit=limit)

    return TrainingJobListResponse(
        jobs=[
            TrainingJobSummary(
                job_id=j.job_id,
                ticker=j.ticker,
                status=j.status,
                created_at=j.created_at,
            )
            for j in jobs
        ],
        total=len(jobs),
    )


@router.get("/models", response_model=ModelListResponse)
def list_models(
        ticker: str = None,
        active_only: bool = False,
        db: Session = Depends(get_database),
):
    """Lista modelos registrados."""
    repo = ModelRegistryRepository()
    models = repo.list_models(db, ticker=ticker, active_only=active_only)

    return ModelListResponse(
        models=[
            ModelInfo(
                version_id=m.version_id,
                ticker=m.ticker,
                mae=m.mae,
                rmse=m.rmse,
                mape=m.mape,
                r2_score=m.r2_score,
                epochs_trained=m.epochs_trained,
                hyperparameters=json.loads(m.hyperparameters) if m.hyperparameters else None,
                is_active=m.is_active,
                created_at=m.created_at,
            )
            for m in models
        ]
    )


@router.post("/activate/{version_id}", response_model=ActivateModelResponse)
def activate_model(version_id: str, db: Session = Depends(get_database)):
    """Ativa um modelo específico."""
    repo = ModelRegistryRepository()

    model = repo.get_model(db, version_id)
    if not model:
        raise HTTPException(404, f"Model {version_id} not found")

    # Buscar modelo ativo anterior
    previous = repo.get_active_model(db, model.ticker)
    previous_id = previous.version_id if previous else None

    # Ativar
    repo.set_active_model(db, model.ticker, version_id)
    db.commit()

    return ActivateModelResponse(
        status="activated",
        version_id=version_id,
        ticker=model.ticker,
        previous_active=previous_id,
    )
