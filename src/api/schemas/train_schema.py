from pydantic import BaseModel, Field, field_validator

class TrainRequest(BaseModel):
    """Schema para requisição de treinamento."""

    ticker: str = Field(..., description="Símbolo da ação para treinar")
    epochs: int = Field(default=100, ge=10, le=500)
    batch_size: int = Field(default=32, ge=8, le=128)
    sequence_length: int = Field(default=60, ge=10, le=120)
    train_split: float = Field(default=0.7, ge=0.5, le=0.9)
    val_split: float = Field(default=0.15, ge=0.05, le=0.3)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        return v.upper().strip()