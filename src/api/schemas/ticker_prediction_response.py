from pydantic import Field


class TickerPredictionResponse(PredictionResponse):
    """Schema para resposta de previsão por ticker."""

    ticker: str = Field(..., description="Símbolo da ação")