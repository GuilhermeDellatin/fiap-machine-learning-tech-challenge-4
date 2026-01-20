from pydantic import BaseModel, Field, field_validator
from typing import List

class PredictionRequest(BaseModel):
    """Schema para requisição de previsão."""

    prices: List[float] = Field(
        ...,
        description="Lista de preços históricos de fechamento",
        min_length=60,
        examples=[[150.0, 151.2, 149.8, 152.5, 153.0]]
    )
    days_ahead: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Número de dias para prever (1-30)"
    )

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v):
        if any(p <= 0 for p in v):
            raise ValueError("Todos os preços devem ser positivos")
        return v