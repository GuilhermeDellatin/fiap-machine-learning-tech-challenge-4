"""
Schemas Pydantic para treinamento.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class TrainingRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=8, le=256)
    sequence_length: int = Field(default=60, ge=10, le=200)
    hidden_size: int = Field(default=64, ge=16, le=512)
    num_layers: int = Field(default=2, ge=1, le=5)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1)

    @field_validator('ticker')
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    ticker: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    progress_percent: float
    epochs_completed: int
    epochs_total: int
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    error_message: Optional[str] = None
    model_version_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrainingJobSummary(BaseModel):
    job_id: str
    ticker: str
    status: str
    created_at: datetime


class TrainingJobListResponse(BaseModel):
    jobs: List[TrainingJobSummary]
    total: int


class ModelInfo(BaseModel):
    version_id: str
    ticker: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2_score: Optional[float] = None
    epochs_trained: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    is_active: bool
    created_at: datetime


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ActivateModelResponse(BaseModel):
    status: str
    version_id: str
    ticker: str
    previous_active: Optional[str] = None
