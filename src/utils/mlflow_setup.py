"""
Utilitários para configuração do MLflow.

Este módulo centraliza a configuração do MLflow para evitar
duplicação de código entre CLI e API.
"""
import mlflow
from typing import Optional
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_mlflow_configured = False


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> str:
    """
    Configura MLflow tracking.

    Deve ser chamado UMA VEZ no início do orquestrador.
    Chamadas subsequentes retornam o experiment_id sem reconfigurar.

    Args:
        tracking_uri: URI do tracking server (default: settings)
        experiment_name: Nome do experimento (default: settings)

    Returns:
        experiment_id criado/recuperado

    Example:
        >>> experiment_id = setup_mlflow()
        >>> with mlflow.start_run(run_name="AAPL_20240115"):
        ...     # treinar modelo
    """
    global _mlflow_configured

    # Evitar reconfiguração desnecessária
    if _mlflow_configured:
        exp_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            return experiment.experiment_id

    uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    exp_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)

    # Criar ou recuperar experimento
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            exp_name,
            artifact_location=settings.MLFLOW_ARTIFACT_ROOT
        )
        logger.info(f"Experimento MLflow criado: {exp_name} (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Usando experimento existente: {exp_name} (id={experiment_id})")

    mlflow.set_experiment(exp_name)
    _mlflow_configured = True

    return experiment_id


def generate_version_id(ticker: str) -> str:
    """
    Gera version_id consistente para uso em:
    - Nome do arquivo .pt
    - run_name do MLflow
    - Registro no ModelRegistry

    Args:
        ticker: Código da ação (ex: "PETR4.SA")

    Returns:
        version_id no formato "{ticker}_{timestamp}"

    Example:
        >>> version_id = generate_version_id("AAPL")
        >>> # "AAPL_20240115_143022"
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ticker}_{timestamp}"


def is_mlflow_active() -> bool:
    """
    Verifica se há uma run MLflow ativa.

    Útil para logging condicional no Trainer.

    Returns:
        True se há run ativa, False caso contrário
    """
    return mlflow.active_run() is not None