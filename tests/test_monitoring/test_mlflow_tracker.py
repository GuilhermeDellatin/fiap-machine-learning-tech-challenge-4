"""
Tests for MLflow tracker helper.
"""
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.monitoring.mlflow_tracker import MLflowTracker


class FakeMLflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.run_name = None
        self.params = {}
        self.metrics = []
        self.tags = {}
        self.artifacts = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment = experiment_name

    @contextmanager
    def start_run(self, run_name=None):
        self.run_name = run_name
        yield {"run_name": run_name}

    def active_run(self):
        return {"run_name": self.run_name}

    def set_tags(self, tags: dict) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step=None) -> None:
        self.metrics.append((metrics, step))

    def log_artifact(self, local_path: str, artifact_path=None) -> None:
        self.artifacts.append((local_path, artifact_path))


def test_tracker_disables_when_mlflow_is_missing(monkeypatch):
    def _raise_module_not_found(_):
        raise ModuleNotFoundError

    monkeypatch.setattr(
        "src.monitoring.mlflow_tracker.importlib.import_module",
        _raise_module_not_found,
    )

    tracker = MLflowTracker(
        enabled=True,
        tracking_uri="file:./mlruns",
        experiment_name="test-experiment",
    )

    assert tracker.enabled is False


def test_tracker_noop_when_disabled():
    tracker = MLflowTracker(
        enabled=False,
        tracking_uri="file:./mlruns",
        experiment_name="test-experiment",
    )

    with tracker.start_run(run_name="noop-run"):
        tracker.log_params({"a": 1})
        tracker.log_metrics({"metric": 1.0}, step=1)
        tracker.set_tags({"stage": "test"})


def test_tracker_logs_with_fake_mlflow(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(
        "src.monitoring.mlflow_tracker.importlib.import_module",
        lambda _: fake_mlflow,
    )

    artifact = Path("tests_mlflow_artifact.txt")
    artifact.write_text("artifact-data", encoding="utf-8")

    try:
        tracker = MLflowTracker(
            enabled=True,
            tracking_uri="file:./mlruns",
            experiment_name="test-experiment",
        )

        with tracker.start_run(run_name="run-1", tags={"pipeline": "test"}):
            tracker.log_params({"epochs": 5, "metadata": {"x": 1}})
            tracker.log_metrics({"loss": 0.12, "non_numeric": "abc"}, step=2)
            tracker.log_artifact(str(artifact), artifact_path="model")
    finally:
        artifact.unlink(missing_ok=True)

    assert fake_mlflow.tracking_uri == "file:./mlruns"
    assert fake_mlflow.experiment == "test-experiment"
    assert fake_mlflow.run_name == "run-1"
    assert fake_mlflow.tags["pipeline"] == "test"
    assert fake_mlflow.params["epochs"] == 5
    assert isinstance(fake_mlflow.params["metadata"], str)

    logged_metrics, step = fake_mlflow.metrics[0]
    assert logged_metrics["loss"] == pytest.approx(0.12)
    assert "non_numeric" not in logged_metrics
    assert step == 2

    artifact_path, artifact_group = fake_mlflow.artifacts[0]
    assert artifact_path.endswith("artifact.txt")
    assert artifact_group == "model"
