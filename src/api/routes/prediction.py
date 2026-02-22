"""
Endpoints de predição com MLflow tracing.
"""
import time
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
from src.utils.mlflow_tracing import tracing

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(
        request: PredictionRequest,
        db: Session = Depends(get_database),
):
    """
    Predição de preços futuros com MLflow tracing.


    REGRA: Se não houver modelo treinado, retorna 404.
    """
    model_repo = ModelRegistryRepository()
    predictor = get_predictor()
    collector = get_collector()

    with tracing.safe_span(
            "prediction_pipeline",
            inputs={"ticker": request.ticker, "days_ahead": request.days_ahead},
    ) as root_span:

        # 1. Buscar modelo ativo
        with tracing.load_model(request.ticker) as span:
            model_info = model_repo.get_active_model(db, request.ticker)

            if model_info is None:
                span.set_attributes({"error": "no_model_found"})
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": f"No trained model found for ticker {request.ticker}",
                        "suggestion": "Train a model first using POST /api/v1/training/start",
                        "ticker": request.ticker,
                    },
                )

            span.set_outputs({
                "version_id": model_info.version_id,
                "model_path": model_info.model_path,
            })

        # 2. Carregar modelo se necessário
        with tracing.ensure_model_loaded(request.ticker) as span:
            model_reloaded = False
            if not predictor.is_loaded() or predictor.current_ticker != request.ticker:
                start_load = time.time()
                predictor.reload_model(model_info.model_path, model_info.scaler_path)
                predictor.current_ticker = request.ticker
                predictor.model_version = model_info.version_id
                model_reloaded = True
                span.set_attributes({"model_load_time_ms": (time.time() - start_load) * 1000})

            span.set_outputs({
                "model_reloaded": model_reloaded,
                "model_version": model_info.version_id,
            })

        # 3. Buscar dados históricos
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        with tracing.data_collection(request.ticker, start_date=start_date, end_date=end_date) as span:
            df = collector.download_data(db, request.ticker, start_date, end_date)
            span.set_outputs({"data_points": len(df)})
            span.set_attributes({"dataset.rows": len(df)})

        # 4. Fazer predição
        with tracing.prediction(request.days_ahead, len(df)) as span:
            start_pred = time.time()
            predictions = predictor.predict(df, days_ahead=request.days_ahead)
            prediction_time_ms = (time.time() - start_pred) * 1000
            span.set_outputs({
                "predictions_count": len(predictions),
                "prediction_values": [round(p, 2) for p in predictions],
            })
            span.set_attributes({"prediction_time_ms": prediction_time_ms})

        # 5. Formatar resposta
        with tracing.format_response() as span:
            prediction_items = []
            base_date = datetime.now()
            for i, price in enumerate(predictions):
                pred_date = base_date + timedelta(days=i + 1)
                while pred_date.weekday() >= 5:
                    pred_date += timedelta(days=1)
                prediction_items.append(
                    PredictionItem(
                        date=pred_date.strftime("%Y-%m-%d"),
                        price=round(price, 2),
                    )
                )
            span.set_outputs({"items_count": len(prediction_items)})

        if root_span:
            root_span.set_outputs({
                "ticker": request.ticker,
                "model_version": model_info.version_id,
                "predictions_count": len(prediction_items),
                "prediction_time_ms": prediction_time_ms,
            })

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
