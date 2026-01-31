"""
Schemas Pydantic para inferência.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List


class InferenceRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    data: Optional[List[List[float]]] = None
    raw_prices: Optional[List[float]] = None
    return_normalized: bool = False

    @model_validator(mode='after')
    def check_data_or_prices(self):
        if self.data is None and self.raw_prices is None:
            raise ValueError("Must provide 'data' or 'raw_prices'")
        if self.data is not None and self.raw_prices is not None:
            raise ValueError("Provide only one: 'data' or 'raw_prices'")
        return self


class InferenceResponse(BaseModel):
    ticker: str
    prediction: float
    is_normalized: bool
    model_version: str
    inference_time_ms: float


class BatchInferenceRequest(BaseModel):
    ticker: str
    sequences: List[List[List[float]]]


class BatchInferenceResponse(BaseModel):
    ticker: str
    predictions: List[float]
    batch_size: int
    inference_time_ms: float


class WarmupResponse(BaseModel):
    status: str
    inference_time_ms: float
    device: str
