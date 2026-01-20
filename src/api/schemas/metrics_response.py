from pydantic import BaseModel, Field

class MetricsResponse(BaseModel):
    """Schema para métricas do modelo."""

    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2_score: float = Field(..., description="Coeficiente R²")

    class Config:
        json_schema_extra = {
            "example": {
                "mae": 2.34,
                "rmse": 3.12,
                "mape": 1.56,
                "r2_score": 0.94
            }
        }