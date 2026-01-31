"""
Endpoints de predição.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_database, get_collector, get_predictor
from src.database.repository import ModelRegistryRepository
from src.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FailedPrediction,
)
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_database),
):
    """
    Predição de preços futuros.

    REGRA: Se não houver modelo treinado, retorna 404.
    """
    model_repo = ModelRegistryRepository()
    predictor = get_predictor()
    collector = get_collector()

    # 1. Buscar modelo ativo
    model_info = model_repo.get_active_model(db, request.ticker)

    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No trained model found for ticker {request.ticker}",
                "suggestion": "Train a model first using POST /api/v1/training/start",
                "ticker": request.ticker,
            },
        )

    # 2. Carregar modelo se necessário
    if not predictor.is_loaded() or predictor.current_ticker != request.ticker:
        predictor.reload_model(model_info.model_path, model_info.scaler_path)
        predictor.current_ticker = request.ticker
        predictor.model_version = model_info.version_id

    # 3. Buscar dados históricos
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    df = collector.download_data(db, request.ticker, start_date, end_date)

    # 4. Fazer predição
    predictions = predictor.predict(df, days_ahead=request.days_ahead)

    # 5. Formatar resposta
    prediction_items = []
    base_date = datetime.now()
    for i, price in enumerate(predictions):
        pred_date = base_date + timedelta(days=i + 1)
        # Pular fins de semana
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)
        prediction_items.append(
            PredictionItem(
                date=pred_date.strftime("%Y-%m-%d"),
                price=round(price, 2),
            )
        )

    return PredictionResponse(
        ticker=request.ticker,
        model_version=model_info.version_id,
        predictions=prediction_items,
        generated_at=datetime.utcnow(),
    )


@router.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_get(
    ticker: str,
    days_ahead: int = 1,
    db: Session = Depends(get_database),
):
    """Predição via GET."""
    request = PredictionRequest(ticker=ticker, days_ahead=days_ahead)
    return predict(request, db)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_database),
):
    """Predição em lote."""
    results = {}
    failed = []

    for ticker in request.tickers:
        try:
            pred_request = PredictionRequest(ticker=ticker, days_ahead=request.days_ahead)
            result = predict(pred_request, db)
            results[ticker] = result
        except HTTPException as e:
            failed.append(FailedPrediction(ticker=ticker, error=str(e.detail)))
        except Exception as e:
            failed.append(FailedPrediction(ticker=ticker, error=str(e)))

    return BatchPredictionResponse(
        predictions=results,
        failed=failed,
        generated_at=datetime.utcnow(),
    )
