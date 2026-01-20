from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date

class HistoricalDataRequest(BaseModel):
    """Schema para requisição de dados históricos."""

    ticker: str = Field(..., description="Símbolo da ação")
    start_date: Optional[date] = Field(
        default=None,
        description="Data inicial (YYYY-MM-DD)"
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Data final (YYYY-MM-DD)"
    )
    period: str = Field(
        default="1y",
        description="Período alternativo (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        return v.upper().strip()
