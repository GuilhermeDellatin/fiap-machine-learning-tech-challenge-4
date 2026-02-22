"""
Endpoints de inferência direta com MLflow tracing.
"""
import time

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_database, get_predictor
from src.database.repository import ModelRegistryRepository
from src.api.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    WarmupResponse,
)
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.mlflow_tracing import tracing

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=InferenceResponse)
def inference(
        request: InferenceRequest,
        db: Session = Depends(get_database),
):
    """Inferência direta com MLflow tracing."""
    predictor = get_predictor()
    model_repo = ModelRegistryRepository()

    with tracing.safe_span(
            "inference_pipeline",
            inputs={
                "ticker": request.ticker,
                "has_raw_prices": request.raw_prices is not None,
                "return_normalized": request.return_normalized,
                "data_length": len(request.data) if request.data else (
                        len(request.raw_prices) if request.raw_prices else 0
                ),
            },
    ) as root_span:

        # Carregar modelo se necessário
        with tracing.load_model(request.ticker.upper()) as span:
            model_info = model_repo.get_active_model(db, request.ticker.upper())
            if not model_info:
                span.set_attributes({"error": "no_model_found"})
                raise HTTPException(404, f"No model for {request.ticker}")

            model_reloaded = False
            if not predictor.is_loaded() or predictor.current_ticker != request.ticker.upper():
                start_load = time.time()
                predictor.reload_model(model_info.model_path, model_info.scaler_path)
                predictor.current_ticker = request.ticker.upper()
                predictor.model_version = model_info.version_id
                model_reloaded = True
                span.set_attributes({"model_load_time_ms": (time.time() - start_load) * 1000})

            span.set_outputs({
                "model_version": model_info.version_id,
                "model_reloaded": model_reloaded,
            })

        # Inferência
        if request.raw_prices:
            with tracing.inference("raw_prices", prices_count=len(request.raw_prices)) as span:
                start_time = time.time()
                df = pd.DataFrame({"Close": request.raw_prices})
                predictions = predictor.predict(df, days_ahead=1)
                prediction = predictions[0]
                is_normalized = False
                inference_time = (time.time() - start_time) * 1000
                span.set_outputs({
                    "prediction": prediction,
                    "is_normalized": is_normalized,
                    "inference_time_ms": round(inference_time, 2),
                })
                span.set_attributes({"inference_time_ms": inference_time})
        else:
            with tracing.inference(
                    "normalized_data",
                    sequence_length=len(request.data),
                    return_normalized=request.return_normalized,
            ) as span:
                start_time = time.time()
                input_tensor = torch.FloatTensor(request.data).unsqueeze(0)
                output = predictor.inference(input_tensor)
                prediction = float(output.cpu().numpy()[0][0])
                is_normalized = request.return_normalized

                if not request.return_normalized:
                    prediction = float(
                        predictor.preprocessor.inverse_transform(np.array([[prediction]]))[0]
                    )

                inference_time = (time.time() - start_time) * 1000
                span.set_outputs({
                    "prediction": prediction,
                    "is_normalized": is_normalized,
                    "inference_time_ms": round(inference_time, 2),
                })
                span.set_attributes({"inference_time_ms": inference_time})

        if root_span:
            root_span.set_outputs({
                "ticker": request.ticker.upper(),
                "prediction": prediction,
                "is_normalized": is_normalized,
                "model_version": predictor.model_version,
                "inference_time_ms": round(inference_time, 2),
            })

    return InferenceResponse(
        ticker=request.ticker.upper(),
        prediction=prediction,
        is_normalized=is_normalized,
        model_version=predictor.model_version,
        inference_time_ms=round(inference_time, 2),
    )


@router.post("/batch", response_model=BatchInferenceResponse)
def batch_inference(
        request: BatchInferenceRequest,
        db: Session = Depends(get_database),
):
    """Batch inference com MLflow tracing."""
    predictor = get_predictor()
    model_repo = ModelRegistryRepository()

    with tracing.safe_span(
            "batch_inference_pipeline",
            inputs={"ticker": request.ticker.upper(), "batch_size": len(request.sequences)},
    ) as root_span:

        with tracing.load_model(request.ticker.upper()) as span:
            model_info = model_repo.get_active_model(db, request.ticker.upper())
            if not model_info:
                span.set_attributes({"error": "no_model_found"})
                raise HTTPException(404, f"No model for {request.ticker}")

            if not predictor.is_loaded():
                predictor.reload_model(model_info.model_path, model_info.scaler_path)

            span.set_outputs({"model_version": model_info.version_id})

        with tracing.batch_inference(len(request.sequences)) as span:
            start_time = time.time()
            input_tensor = torch.FloatTensor(request.sequences)
            output = predictor.inference(input_tensor)
            predictions = output.cpu().numpy().flatten().tolist()
            inference_time = (time.time() - start_time) * 1000
            span.set_outputs({
                "predictions_count": len(predictions),
                "inference_time_ms": round(inference_time, 2),
            })
            span.set_attributes({"inference_time_ms": inference_time})

        if root_span:
            root_span.set_outputs({
                "ticker": request.ticker.upper(),
                "predictions_count": len(predictions),
                "batch_size": len(request.sequences),
                "inference_time_ms": round(inference_time, 2),
            })

    return BatchInferenceResponse(
        ticker=request.ticker.upper(),
        predictions=predictions,
        batch_size=len(request.sequences),
        inference_time_ms=round(inference_time, 2),
    )


@router.get("/warmup", response_model=WarmupResponse)
def warmup(db: Session = Depends(get_database)):
    """Aquece o modelo."""
    predictor = get_predictor()

    if not predictor.is_loaded():
        return WarmupResponse(
            status="no_model_loaded",
            inference_time_ms=0,
            device="unknown",
        )

    # Very Simple inference
    start_time = time.time()
    warmup_input = torch.randn(1, settings.SEQUENCE_LENGTH, 1)
    predictor.inference(warmup_input)
    inference_time = (time.time() - start_time) * 1000

    return WarmupResponse(
        status="warmed_up",
        inference_time_ms=round(inference_time, 2),
        device=str(predictor.device),
    )
