"""
Middleware de métricas HTTP.
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.monitoring.metrics import record_request


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware para coletar métricas de requests."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Não registrar /metrics para evitar loop
        if request.url.path != "/metrics":
            record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )

        return response
