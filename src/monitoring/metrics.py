"""
Métricas Prometheus centralizadas.
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# Registry singleton
REGISTRY = CollectorRegistry()

# HTTP Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Model Metrics
MODEL_PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['ticker'],
    registry=REGISTRY
)

MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['ticker'],
    registry=REGISTRY
)

# Training Metrics
TRAINING_JOBS_TOTAL = Counter(
    'training_jobs_total',
    'Total training jobs',
    ['status'],
    registry=REGISTRY
)

TRAINING_JOBS_ACTIVE = Gauge(
    'training_jobs_active',
    'Active training jobs',
    registry=REGISTRY
)

# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    'cache_hits_total',
    'Cache hits',
    ['ticker'],
    registry=REGISTRY
)

CACHE_MISSES_TOTAL = Counter(
    'cache_misses_total',
    'Cache misses',
    ['ticker'],
    registry=REGISTRY
)


def get_metrics() -> bytes:
    """Retorna métricas no formato Prometheus."""
    return generate_latest(REGISTRY)


# Helper functions
def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Registra métricas de request."""
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def record_prediction(ticker: str, duration: float) -> None:
    """Registra métricas de predição."""
    MODEL_PREDICTIONS_TOTAL.labels(ticker=ticker).inc()
    MODEL_INFERENCE_DURATION.labels(ticker=ticker).observe(duration)


def record_cache_hit(ticker: str) -> None:
    """Registra cache hit."""
    CACHE_HITS_TOTAL.labels(ticker=ticker).inc()


def record_cache_miss(ticker: str) -> None:
    """Registra cache miss."""
    CACHE_MISSES_TOTAL.labels(ticker=ticker).inc()


def record_training_job(status: str) -> None:
    """Registra job de treinamento."""
    TRAINING_JOBS_TOTAL.labels(status=status).inc()


def set_active_jobs(count: int) -> None:
    """Define número de jobs ativos."""
    TRAINING_JOBS_ACTIVE.set(count)
