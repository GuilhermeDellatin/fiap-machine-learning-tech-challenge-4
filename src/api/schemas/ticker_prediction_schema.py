from pydantic import BaseModel, Field, field_validator

class TickerPredictionRequest(BaseModel):
    """Schema para previsão usando ticker."""

    ticker: str = Field(
        ...,
        description="Símbolo da ação (ex: AAPL, GOOGL, PETR4.SA)",
        min_length=1,
        max_length=10,
        examples=["AAPL"]
    )
    days_ahead: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Número de dias para prever"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        return v.upper().strip()