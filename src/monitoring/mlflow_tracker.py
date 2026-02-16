"""
MLflow tracking helper with graceful fallback behavior.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """Light wrapper around MLflow used by training/evaluation pipelines."""

    def __init__(self, enabled: bool, tracking_uri: str, experiment_name: str) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow: Any | None = None

        if not self.enabled:
            logger.info("MLflow tracking disabled by configuration.")
            return

        try:
            self._mlflow = importlib.import_module("mlflow")
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            logger.info(
                "MLflow enabled | uri=%s | experiment=%s",
                self.tracking_uri,
                self.experiment_name,
            )
        except ModuleNotFoundError:
            self.enabled = False
            logger.warning("MLflow package not found. Tracking disabled.")
        except Exception as exc:  # pragma: no cover - defensive guard
            self.enabled = False
            logger.warning("Failed to initialize MLflow. Tracking disabled: %s", exc)

    @contextmanager
    def start_run(
        self, run_name: str | None = None, tags: dict[str, Any] | None = None
    ) -> Iterator[Any | None]:
        """Start MLflow run when enabled. Behaves as no-op otherwise."""
        if not self.enabled or self._mlflow is None:
            yield None
            return

        try:
            with self._mlflow.start_run(run_name=run_name):
                if tags:
                    self.set_tags(tags)
                yield self._mlflow.active_run()
        except Exception as exc:
            logger.warning("Failed to start MLflow run: %s", exc)
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.enabled or self._mlflow is None or not params:
            return

        try:
            safe_params = {k: self._to_param_value(v) for k, v in params.items()}
            self._mlflow.log_params(safe_params)
        except Exception as exc:
            logger.warning("Failed to log MLflow params: %s", exc)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log numeric metrics to MLflow."""
        if not self.enabled or self._mlflow is None or not metrics:
            return

        safe_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                safe_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue

        if not safe_metrics:
            return

        try:
            if step is None:
                self._mlflow.log_metrics(safe_metrics)
            else:
                self._mlflow.log_metrics(safe_metrics, step=step)
        except Exception as exc:
            logger.warning("Failed to log MLflow metrics: %s", exc)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set tags in current MLflow run."""
        if not self.enabled or self._mlflow is None or not tags:
            return

        try:
            safe_tags = {key: str(value) for key, value in tags.items()}
            self._mlflow.set_tags(safe_tags)
        except Exception as exc:
            logger.warning("Failed to set MLflow tags: %s", exc)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log single artifact file to MLflow if it exists."""
        if not self.enabled or self._mlflow is None:
            return

        path = Path(local_path)
        if not path.exists():
            logger.warning("Artifact not found, skipping MLflow upload: %s", local_path)
            return

        try:
            if artifact_path:
                self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
            else:
                self._mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.warning("Failed to log MLflow artifact: %s", exc)

    @staticmethod
    def _to_param_value(value: Any) -> str | int | float | bool:
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
