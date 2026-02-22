"""
MLflow setup helpers.

This module centralizes MLflow setup to avoid duplicated logic in CLI and API.
"""
from typing import Optional

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_mlflow_configured = False

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    _MLFLOW_AVAILABLE = False


def setup_mlflow(
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
) -> str:
    """
    Configure MLflow tracking and experiment.

    Must be called once by the orchestrator.
    Subsequent calls return the same experiment id.
    """
    if not _MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow package is not installed in this environment")

    global _mlflow_configured

    if _mlflow_configured:
        exp_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            return experiment.experiment_id

    uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    exp_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)

    try:
        mlflow.tracing.enable()
        logger.info("MLflow tracing habilitado")
    except Exception as e:
        logger.warning(f"Falha ao habilitar MLflow tracing: {e}")

    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            exp_name,
            artifact_location=settings.MLFLOW_ARTIFACT_ROOT,
        )
        logger.info(f"MLflow experiment created: {exp_name} (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {exp_name} (id={experiment_id})")

    mlflow.set_experiment(exp_name)
    _mlflow_configured = True

    return experiment_id


def generate_version_id(ticker: str) -> str:
    """Generate a consistent version id for model and run naming."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ticker}_{timestamp}"


def is_mlflow_active() -> bool:
    """Return True when MLflow is available and an active run exists."""
    if not _MLFLOW_AVAILABLE:
        return False

    return mlflow.active_run() is not None
