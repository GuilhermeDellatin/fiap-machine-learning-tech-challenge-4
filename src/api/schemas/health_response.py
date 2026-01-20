from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class HealthResponse(BaseModel):
    """Schema para resposta de health check."""

    status: str = Field(
        ...,
        description="Status do serviço (healthy/unhealthy)"
    )
    model_loaded: bool = Field(
        ...,
        description="Se o modelo está carregado"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    version: str = Field(
        ...,
        description="Versão da API"
    )
    uptime_seconds: Optional[float] = None