"""
Endpoints de treinamento.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
import torch

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from src.api.dependencies import get_database, get_collector
from src.database.connection import SessionLocal
from src.database.repository import (
    TrainingJobRepository,
    ModelRegistryRepository,
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

router = APIRouter()
logger = get_logger(__name__)


def train_model_task(job_id: str, ticker: str, params: dict) -> None:
    """
    Background task para treinamento.

    IMPORTANTE: Esta função roda em background e deve:
    1. Atualizar status do job no banco
    2. Sincronizar dados
    3. Treinar modelo
    4. Salvar .pt e .joblib
    5. Registrar no ModelRegistry
    6. Ativar modelo
    """
    db = SessionLocal()
    job_repo = TrainingJobRepository()
    model_repo = ModelRegistryRepository()

    try:
        # 1. Status: running
        job_repo.update_job(db, job_id, status="running", started_at=datetime.utcnow())
        db.commit()

        # 2. Coletar dados
        collector = StockDataCollector()
        df = collector.sync_data(db, ticker)

        # 3. Preprocessar
        preprocessor = DataPreprocessor()
        scaled = preprocessor.fit_transform(df)
        X, y = preprocessor.create_sequences(scaled, params["sequence_length"])
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)

        # 4. Criar modelo
        model = LSTMPredictor(
            input_size=1,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            torch.zeros(1).to(device)
        except Exception:
            device = torch.device("cpu")
        model = model.to(device)

        # 5. Treinar
        trainer = ModelTrainer(model, learning_rate=params["learning_rate"])

        train_loader = trainer.create_dataloader(X_train, y_train, params["batch_size"])
        val_loader = trainer.create_dataloader(X_val, y_val, params["batch_size"], shuffle=False)
        test_loader = trainer.create_dataloader(X_test, y_test, params["batch_size"], shuffle=False)

        def progress_callback(epoch, total_epochs, train_loss, val_loss, best_loss):
            job_repo.update_job(
                db, job_id,
                epochs_completed=epoch,
                current_loss=val_loss,
                best_loss=best_loss,
            )
            db.commit()

        history = trainer.train(train_loader, val_loader, params["epochs"], progress_callback)

        # 6. Avaliar
        metrics = trainer.evaluate(test_loader)

        # 7. Salvar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{ticker}_{timestamp}"
        model_path = f"{settings.MODEL_DIR}/{version_id}.pt"
        scaler_path = f"{settings.MODEL_DIR}/{version_id}_scaler.joblib"

        Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_path)
        preprocessor.save_scaler(scaler_path)

        # 8. Registrar modelo
        registered = model_repo.register_model(
            db, ticker, model_path, scaler_path,
            metrics, params, history.get("best_epoch", params["epochs"]),
        )

        # 9. Ativar modelo
        model_repo.set_active_model(db, ticker, registered.version_id)

        # 10. Job completed
        job_repo.update_job(
            db, job_id,
            status="completed",
            completed_at=datetime.utcnow(),
            model_version_id=registered.version_id,
        )
        db.commit()

        logger.info(f"Training completed: {registered.version_id}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        job_repo.update_job(
            db, job_id,
            status="failed",
            error_message=str(e),
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
    Inicia treinamento em background.
    Retorna 202 Accepted imediatamente.
    """
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
        message=f"Training job started for {request.ticker}",
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
