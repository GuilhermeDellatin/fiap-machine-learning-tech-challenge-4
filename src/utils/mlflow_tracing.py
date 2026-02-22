"""
Abstração reutilizável para MLflow Tracing.


Centraliza toda a lógica de `mlflow.start_span` em context managers
semânticos, evitando duplicação nos endpoints e background tasks.


Uso:
   from src.utils.mlflow_tracing import tracing


   with tracing.pipeline("inference_pipeline", inputs={...}) as root:
       with tracing.load_model(ticker) as span:
           span.set_outputs({"version": "v1"})
       with tracing.inference(mode="raw_prices") as span:
           span.set_outputs({"prediction": 42.0})
"""
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tipo de retorno dos context managers (span ou None no modo safe)
_Span = Any


class MLflowTracing:
    """
    Wrapper em torno de `mlflow.start_span` que:


    - Expõe context managers genéricos (`span`, `pipeline`).
    - Expõe context managers semânticos para cada etapa do domínio
      (`load_model`, `inference`, `prediction`, `data_collection`, …).
    - Oferece `safe_span`: se o MLflow não estiver disponível o bloco
      ainda é executado normalmente (fallback silencioso).
    """

    # ─────────────────────────────────────────────────────────────
    # Primitivos genéricos
    # ─────────────────────────────────────────────────────────────

    @contextmanager
    def span(
            self,
            name: str,
            inputs: Optional[dict[str, Any]] = None,
            outputs: Optional[dict[str, Any]] = None,
            attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[_Span]:
        """
        Context manager genérico em torno de `mlflow.start_span`.


        Propaga exceções normalmente (exceto falhas internas do MLflow).
        """
        with mlflow.start_span(name=name) as s:
            if inputs:
                s.set_inputs(inputs)
            if attributes:
                s.set_attributes(attributes)
            yield s
            if outputs:
                s.set_outputs(outputs)

    @contextmanager
    def pipeline(
            self,
            name: str,
            inputs: Optional[dict[str, Any]] = None,
    ) -> Iterator[_Span]:
        """
        Span raiz de um pipeline (nível mais alto do trace).
        Use `span.set_outputs(...)` dentro do bloco para registrar saídas.
        """
        with self.span(name=name, inputs=inputs) as root:
            yield root

    @contextmanager
    def safe_span(
            self,
            name: str,
            inputs: Optional[dict[str, Any]] = None,
    ) -> Iterator[Optional[_Span]]:
        """
        Igual a `span`, mas absorve falhas do MLflow silenciosamente.


        O bloco é sempre executado; se o tracing falhar, `span` será
        `None` e os `set_inputs / set_outputs` serão no-ops.
        Ideal para envolver blocos de lógica de negócio que não devem
        quebrar por causa de problemas de observabilidade.


        Exemplo:
            with tracing.safe_span("inference_pipeline", inputs={...}) as root:
                result = do_heavy_work()      # nunca falha por causa do MLflow
                if root:
                    root.set_outputs({"result": result})
        """
        try:
            with self.span(name=name, inputs=inputs) as s:
                yield s
        except Exception as exc:
            logger.warning(f"[MLflowTracing] span '{name}' ignorado: {exc}")
            yield None

    # ─────────────────────────────────────────────────────────────
    # Spans semânticos – nomes e inputs padronizados por domínio
    # ─────────────────────────────────────────────────────────────

    @contextmanager
    def load_model(self, ticker: str) -> Iterator[_Span]:
        """Span para busca e carregamento de modelo ativo."""
        with self.span("load_model", inputs={"ticker": ticker}) as s:
            yield s

    @contextmanager
    def ensure_model_loaded(self, ticker: str) -> Iterator[_Span]:
        """Span para garantir que o modelo está carregado em memória."""
        with self.span("ensure_model_loaded", inputs={"ticker": ticker}) as s:
            yield s

    @contextmanager
    def inference(self, mode: str, **extra_inputs) -> Iterator[_Span]:
        """
        Span para etapa de inferência.


        Args:
            mode: "raw_prices" | "normalized_data" | "batch"
            **extra_inputs: quaisquer outros campos a registrar em set_inputs
        """
        inputs = {"mode": mode, **extra_inputs}
        with self.span("model_inference", inputs=inputs) as s:
            yield s

    @contextmanager
    def batch_inference(self, batch_size: int) -> Iterator[_Span]:
        """Span para etapa de batch inference."""
        with self.span("batch_model_inference", inputs={"batch_size": batch_size}) as s:
            yield s

    @contextmanager
    def prediction(self, days_ahead: int, input_data_points: int) -> Iterator[_Span]:
        """Span para etapa de predição futura."""
        with self.span(
                "model_predict",
                inputs={"days_ahead": days_ahead, "input_data_points": input_data_points},
        ) as s:
            yield s

    @contextmanager
    def data_collection(self, ticker: str, **extra_inputs) -> Iterator[_Span]:
        """Span para etapa de coleta / download de dados históricos."""
        inputs = {"ticker": ticker, **extra_inputs}
        with self.span("fetch_historical_data", inputs=inputs) as s:
            yield s

    @contextmanager
    def preprocessing(self, sequence_length: int, dataset_size: int) -> Iterator[_Span]:
        """Span para etapa de pré-processamento."""
        with self.span(
                "preprocessing",
                inputs={"sequence_length": sequence_length, "dataset_size": dataset_size},
        ) as s:
            yield s

    @contextmanager
    def model_creation(self, **hyperparams) -> Iterator[_Span]:
        """Span para criação do modelo LSTM."""
        with self.span("model_creation", inputs=hyperparams) as s:
            yield s

    @contextmanager
    def model_training(self, epochs: int, learning_rate: float, batch_size: int) -> Iterator[_Span]:
        """Span para loop de treinamento."""
        with self.span(
                "model_training",
                inputs={"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size},
        ) as s:
            yield s

    @contextmanager
    def model_evaluation(self, test_size: int) -> Iterator[_Span]:
        """Span para etapa de avaliação no conjunto de teste."""
        with self.span("model_evaluation", inputs={"test_size": test_size}) as s:
            yield s

    @contextmanager
    def save_model(self, model_path: str, scaler_path: str) -> Iterator[_Span]:
        """Span para persistência do modelo e scaler no disco."""
        with self.span(
                "save_model",
                inputs={"model_path": model_path, "scaler_path": scaler_path},
        ) as s:
            yield s

    @contextmanager
    def artifact_logging(self, model_path: str, scaler_path: str) -> Iterator[_Span]:
        """Span para registro de artefatos no MLflow."""
        with self.span(
                "artifact_logging",
                inputs={"model_path": model_path, "scaler_path": scaler_path},
        ) as s:
            yield s

    @contextmanager
    def model_registration(self, ticker: str, version_id: str) -> Iterator[_Span]:
        """Span para registro do modelo no ModelRegistry (SQLite)."""
        with self.span(
                "model_registration",
                inputs={"ticker": ticker, "version_id": version_id},
        ) as s:
            yield s

    @contextmanager
    def format_response(self) -> Iterator[_Span]:
        """Span para formatação da resposta final."""
        with self.span("format_response") as s:
            yield s


# ---------------------------------------------------------------------------
# Instância global – importe diretamente em outros módulos:
#   from src.utils.mlflow_tracing import tracing
# ---------------------------------------------------------------------------
tracing = MLflowTracing()


# ---------------------------------------------------------------------------
# Funções de módulo convenientes para uso direto sem instância
# ---------------------------------------------------------------------------

@contextmanager
def trace_span(
        name: str,
        attributes: Optional[dict[str, Any]] = None,
) -> Iterator[Optional[_Span]]:
    """
    Context manager de span seguro para uso fora da classe MLflowTracing.

    Absorve falhas do MLflow silenciosamente — o bloco é sempre executado.
    Yields None quando o tracing não está disponível.

    Exemplo:
        with trace_span("collector.download_data", attributes={"ticker": ticker}) as span:
            set_span_attribute(span, "cache_hit", "true")
    """
    try:
        with mlflow.start_span(name=name) as s:
            if attributes:
                s.set_attributes(attributes)
            yield s
    except Exception as exc:
        logger.debug(f"[trace_span] span '{name}' ignorado: {exc}")
        yield None


def set_span_attribute(span: Optional[_Span], key: str, value: Any) -> None:
    """
    Define um atributo no span de forma segura (no-op se span for None).

    Exemplo:
        set_span_attribute(span, "cache_hit", "true")
    """
    if span is None:
        return
    try:
        span.set_attributes({key: value})
    except Exception as exc:
        logger.debug(f"[set_span_attribute] falha ao setar '{key}': {exc}")
