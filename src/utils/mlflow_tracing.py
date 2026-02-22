"""
MLflow Tracing utilities — optional, non-breaking.

Fornece spans distribuídos para observabilidade de cada request e etapas internas,
independente do rastreamento de experimentos (mlflow.start_run).

Padrão de uso:
    from src.utils.mlflow_tracing import trace_span, set_span_attribute, set_span_error

    with trace_span("my_operation", attributes={"ticker": "AAPL"}) as span:
        result = do_work()
        set_span_attribute(span, "result_size", str(len(result)))
"""
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


@contextmanager
def trace_span(
    name: str,
    span_type: str = "UNKNOWN",
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager que cria um span MLflow se disponível.

    Fallback seguro (no-op, yield None) quando:
    - MLflow não está instalado
    - Criação do span falha por qualquer motivo
    - Atributos inválidos

    Args:
        name: Nome do span (ex: "predict_request", "collector.download_data")
        span_type: Tipo do span — "CHAIN", "UNKNOWN", etc.
        attributes: Dict de atributos key-value anexados ao span.
                    Valores são convertidos para str automaticamente.

    Yields:
        span: Objeto Span do MLflow, ou None se não disponível.
    """
    if not _MLFLOW_AVAILABLE:
        yield None
        return

    try:
        with mlflow.start_span(name=name, span_type=span_type) as span:
            if attributes and span is not None:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, str(value))
                    except Exception:
                        pass
            yield span
    except Exception:
        yield None


def set_span_attribute(span: Any, key: str, value: Any) -> None:
    """
    Define um atributo no span de forma segura.

    Não lança exceção se span for None ou se a operação falhar.

    Args:
        span: Objeto Span MLflow (pode ser None).
        key: Nome do atributo.
        value: Valor do atributo (convertido para str).
    """
    if span is None:
        return
    try:
        span.set_attribute(key, str(value))
    except Exception:
        pass


def set_span_error(span: Any, error: str) -> None:
    """
    Marca o span como erro com mensagem descritiva.

    Não lança exceção se span for None ou se a operação falhar.

    Args:
        span: Objeto Span MLflow (pode ser None).
        error: Mensagem de erro (truncada a 500 chars).
    """
    if span is None:
        return
    try:
        span.set_attribute("error.message", str(error)[:500])
        span.set_status("ERROR")
    except Exception:
        try:
            # Fallback: apenas atributo, ignora set_status
            span.set_attribute("error.message", str(error)[:500])
        except Exception:
            pass
