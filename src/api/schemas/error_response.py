from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ErrorResponse(BaseModel):
    """Schema para respostas de erro."""

    success: bool = False
    error: str = Field(..., description="Mensagem de erro")
    error_code: str = Field(..., description="Código do erro")
    details: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Dados insuficientes para previsão",
                "error_code": "INSUFFICIENT_DATA",
                "details": {"required": 60, "provided": 30}
            }
        }