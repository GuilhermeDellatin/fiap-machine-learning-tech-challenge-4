from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    """Schema para resposta de previsão."""

    success: bool = True
    predictions: List[float] = Field(
        ...,
        description="Lista de preços previstos"
    )
    price_changes_percent: List[float] = Field(
        ...,
        description="Variação percentual em relação ao último preço conhecido"
    )
    last_known_price: float = Field(
        ...,
        description="Último preço conhecido usado como referência"
    )
    days_ahead: int = Field(
        ...,
        description="Número de dias previstos"
    )
    generated_at: str = Field(
        ...,
        description="Timestamp da previsão"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [152.34, 153.12, 151.89],
                "price_changes_percent": [1.23, 1.75, 0.95],
                "last_known_price": 150.50,
                "days_ahead": 3,
                "generated_at": "2024-01-15T10:30:00"
            }
        }