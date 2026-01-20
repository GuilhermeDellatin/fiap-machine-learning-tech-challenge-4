from fastapi import APIRouter, Depends
from datetime import datetime

from src.api.schemas.response import HealthResponse, ModelInfoResponse
#from src.api.dependencies import get_predictor_dependency
#from src.models.predictor import StockPredictor
#from src.utils.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

# Timestamp de início para calcular uptime
START_TIME = datetime.now()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica o status da API e do modelo"
)
async def health_check(
        predictor: StockPredictor = Depends(get_predictor_dependency)
) -> HealthResponse:
    """
    Endpoint de health check.

    Retorna o status da aplicação, se o modelo está carregado,
    e informações de versão.
    """
    uptime = (datetime.now() - START_TIME).total_seconds()

    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
        version=settings.version,
        uptime_seconds=uptime
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Verifica se a API está pronta para receber requisições"
)
async def readiness_check(
        predictor: StockPredictor = Depends(get_predictor_dependency)
) -> dict:
    """
    Endpoint de readiness para Kubernetes/Docker.

    Retorna 200 se o modelo está carregado e pronto.
    """
    if predictor.is_loaded:
        return {"ready": True}
    return {"ready": False, "reason": "Model not loaded"}


@router.get(
    "/live",
    summary="Liveness Check",
    description="Verifica se a API está viva"
)
async def liveness_check() -> dict:
    """
    Endpoint de liveness para Kubernetes/Docker.

    Simplesmente retorna OK se a aplicação está rodando.
    """
    return {"alive": True}


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Model Info",
    description="Retorna informações sobre o modelo carregado"
)
async def model_info(
        predictor: StockPredictor = Depends(get_predictor_dependency)
) -> ModelInfoResponse:
    """
    Retorna informações detalhadas sobre o modelo.
    """
    if not predictor.is_loaded:
        return ModelInfoResponse(
            model_type="LSTM",
            sequence_length=settings.sequence_length,
            input_features=1,
            total_parameters=0,
            trained_on=None
        )

    model = predictor._model

    return ModelInfoResponse(
        model_type="LSTM",
        sequence_length=settings.sequence_length,
        input_features=1,
        total_parameters=model.count_params(),
        trained_on=None  # Pode ser carregado de metadata
    )