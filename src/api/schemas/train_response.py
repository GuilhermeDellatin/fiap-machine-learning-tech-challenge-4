from pydantic import BaseModel

class TrainResponse(BaseModel):
    """Schema para resposta de treinamento."""

    success: bool = True
    message: str
    metrics: MetricsResponse
    epochs_completed: int
    training_time_seconds: float