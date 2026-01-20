from fastapi import APIRouter, Depends, HTTPException, status
import logging

from src.api.schemas.request import (
    PredictionRequest,
    TickerPredictionRequest,
    HistoricalDataRequest
)
from src.api.schemas.response import (
    PredictionResponse,
    TickerPredictionResponse,
    HistoricalDataResponse,
    ErrorResponse
)
from src.api.dependencies import get_predictor_dependency
from src.models.predictor import StockPredictor
from src.data.collector import StockDataCollector
from src.utils.config import settings

router = APIRouter(prefix="/predictions", tags=["Predictions"])
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Prever Preços",
    description="Realiza previsão de preços a partir de dados históricos fornecidos"
)
async def predict_prices(
        request: PredictionRequest,
        predictor: StockPredictor = Depends(get_predictor_dependency)
) -> PredictionResponse:
    """
    Endpoint principal de previsão.

    Recebe uma lista de preços históricos e retorna previsões
    para os próximos N dias.

    **Requisitos:**
    - Mínimo de 60 preços históricos (ou sequence_length configurado)
    - Preços devem ser positivos
    - days_ahead entre 1 e 30
    """
    try:
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não está carregado"
            )

        # Realiza previsão
        result = predictor.predict(
            prices=request.prices,
            days_ahead=request.days_ahead
        )

        logger.info(f"Previsão realizada: {request.days_ahead} dias")

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )


@router.post(
    "/ticker",
    response_model=TickerPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Prever por Ticker",
    description="Realiza previsão buscando dados automaticamente pelo símbolo da ação"
)
async def predict_by_ticker(
        request: TickerPredictionRequest,
        predictor: StockPredictor = Depends(get_predictor_dependency)
) -> TickerPredictionResponse:
    """
    Endpoint de previsão por ticker.

    Busca automaticamente os dados históricos do Yahoo Finance
    e realiza a previsão.

    **Exemplos de tickers:**
    - Americanas: AAPL, GOOGL, MSFT, AMZN
    - Brasileiras: PETR4.SA, VALE3.SA, ITUB4.SA
    """
    try:
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não está carregado"
            )

        # Coleta dados
        collector = StockDataCollector(request.ticker)

        if not collector.validate_ticker():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker '{request.ticker}' não encontrado"
            )

        # Busca dados históricos
        df = collector.fetch_historical_data(period="6mo")
        prices = df["Close"].tolist()

        if len(prices) < settings.sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dados insuficientes. Necessário: {settings.sequence_length}, "
                       f"Disponível: {len(prices)}"
            )

        # Realiza previsão
        result = predictor.predict(
            prices=prices,
            days_ahead=request.days_ahead
        )

        logger.info(f"Previsão para {request.ticker}: {request.days_ahead} dias")

        return TickerPredictionResponse(
            ticker=request.ticker,
            **result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na previsão por ticker: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )


@router.post(
    "/historical",
    response_model=HistoricalDataResponse,
    summary="Obter Dados Históricos",
    description="Retorna dados históricos de uma ação"
)
async def get_historical_data(
        request: HistoricalDataRequest
) -> HistoricalDataResponse:
    """
    Endpoint para obter dados históricos.

    Útil para visualização e análise antes de fazer previsões.
    """
    try:
        collector = StockDataCollector(request.ticker)

        if request.start_date and request.end_date:
            df = collector.fetch_historical_data(
                start_date=str(request.start_date),
                end_date=str(request.end_date)
            )
        else:
            df = collector.fetch_historical_data(period=request.period)

        # Converte para lista de dicts
        records = df.to_dict(orient="records")

        # Converte datetime para string
        for record in records:
            if "Date" in record:
                record["Date"] = record["Date"].isoformat()

        return HistoricalDataResponse(
            ticker=request.ticker,
            data=records,
            total_records=len(records),
            date_range={
                "start": records[0]["Date"] if records else None,
                "end": records[-1]["Date"] if records else None
            }
        )

    except Exception as e:
        logger.error(f"Erro ao obter dados históricos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter dados: {str(e)}"
        )


@router.get(
    "/quote/{ticker}",
    summary="Cotação Atual",
    description="Retorna a cotação atual de uma ação"
)
async def get_quote(ticker: str) -> dict:
    """
    Retorna a cotação atual de uma ação.
    """
    try:
        collector = StockDataCollector(ticker.upper())
        return collector.get_latest_price()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter cotação: {str(e)}"
        )