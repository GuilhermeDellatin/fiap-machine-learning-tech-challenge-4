"""
Schemas Pydantic para predição.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    days_ahead: int = Field(default=1, ge=1, le=30)

    @field_validator('ticker')
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()


class PredictionItem(BaseModel):
    date: str
    price: float


class PredictionResponse(BaseModel):
    ticker: str
    model_version: str
    predictions: List[PredictionItem]
    generated_at: datetime


class BatchPredictionRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=20)
    days_ahead: int = Field(default=1, ge=1, le=30)


class FailedPrediction(BaseModel):
    ticker: str
    error: str


class BatchPredictionResponse(BaseModel):
    predictions: Dict[str, PredictionResponse]
    failed: List[FailedPrediction]
    generated_at: datetime


class PredictionErrorResponse(BaseModel):
    error: str
    suggestion: Optional[str] = None
    ticker: str


class CacheInfoItem(BaseModel):
    ticker: str
    records_count: int
    date_range: Dict[str, str]
    last_updated: datetime
    is_valid: bool
    expires_in_hours: float


class CacheInfoResponse(BaseModel):
    caches: List[CacheInfoItem]


class CacheSyncResponse(BaseModel):
    ticker: str
    records_updated: int
    message: str
