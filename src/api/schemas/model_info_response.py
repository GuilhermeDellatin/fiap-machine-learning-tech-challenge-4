from pydantic import BaseModel
from typing import Optional

class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo."""

    model_type: str = "LSTM"
    sequence_length: int
    input_features: int
    total_parameters: int
    trained_on: Optional[str] = None
    metrics: Optional[MetricsResponse] = None