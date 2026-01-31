"""
Endpoints de inferência direta.
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

router = APIRouter()


@router.post("", response_model=InferenceResponse)
def inference(
    request: InferenceRequest,
    db: Session = Depends(get_database),
):
    """Inferência direta."""
    predictor = get_predictor()
    model_repo = ModelRegistryRepository()

    # Carregar modelo se necessário
    model_info = model_repo.get_active_model(db, request.ticker.upper())
    if not model_info:
        raise HTTPException(404, f"No model for {request.ticker}")

    if not predictor.is_loaded() or predictor.current_ticker != request.ticker.upper():
        predictor.reload_model(model_info.model_path, model_info.scaler_path)
        predictor.current_ticker = request.ticker.upper()
        predictor.model_version = model_info.version_id

    start_time = time.time()

    if request.raw_prices:
        # Preprocessar preços brutos
        df = pd.DataFrame({"Close": request.raw_prices})
        predictions = predictor.predict(df, days_ahead=1)
        prediction = predictions[0]
        is_normalized = False
    else:
        # Dados já normalizados
        input_tensor = torch.FloatTensor(request.data).unsqueeze(0)
        output = predictor.inference(input_tensor)
        prediction = float(output.cpu().numpy()[0][0])
        is_normalized = request.return_normalized

        if not request.return_normalized:
            prediction = float(
                predictor.preprocessor.inverse_transform(np.array([[prediction]]))[0]
            )

    inference_time = (time.time() - start_time) * 1000

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
    """Batch inference."""
    predictor = get_predictor()
    model_repo = ModelRegistryRepository()

    model_info = model_repo.get_active_model(db, request.ticker.upper())
    if not model_info:
        raise HTTPException(404, f"No model for {request.ticker}")

    if not predictor.is_loaded():
        predictor.reload_model(model_info.model_path, model_info.scaler_path)

    start_time = time.time()

    input_tensor = torch.FloatTensor(request.sequences)
    output = predictor.inference(input_tensor)
    predictions = output.cpu().numpy().flatten().tolist()

    inference_time = (time.time() - start_time) * 1000

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

    # Dummy inference
    start_time = time.time()
    dummy = torch.randn(1, settings.SEQUENCE_LENGTH, 1)
    predictor.inference(dummy)
    inference_time = (time.time() - start_time) * 1000

    return WarmupResponse(
        status="warmed_up",
        inference_time_ms=round(inference_time, 2),
        device=str(predictor.device),
    )
