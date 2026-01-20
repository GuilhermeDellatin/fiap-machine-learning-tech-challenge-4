from pydantic import BaseModel, Field
from typing import List, Dict, Any

class HistoricalDataResponse(BaseModel):
    """Schema para resposta de dados históricos."""

    success: bool = True
    ticker: str
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros históricos"
    )
    total_records: int
    date_range: Dict[str, str]