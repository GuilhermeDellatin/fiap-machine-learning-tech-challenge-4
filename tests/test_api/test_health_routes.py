"""
Testes para rotas de health check e cache.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_health_check(api_client):
    """Health check retorna status healthy."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_metrics_endpoint(api_client):
    """Metrics endpoint retorna texto Prometheus."""
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_root_endpoint(api_client):
    """Root endpoint retorna info da API."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Stock Price Prediction API"
    assert "version" in data


def test_cache_info_empty(api_client):
    """Cache info sem ticker retorna lista vazia."""
    response = api_client.get("/api/v1/cache/info")
    assert response.status_code == 200
    assert response.json()["caches"] == []


def test_cache_info_with_ticker_empty(api_client):
    """Cache info para ticker sem dados retorna lista vazia."""
    response = api_client.get("/api/v1/cache/info?ticker=XXXX")
    assert response.status_code == 200
    assert response.json()["caches"] == []


def test_cache_info_with_data(api_client, test_db):
    """Cache info para ticker com dados retorna informacoes."""
    from datetime import datetime, timedelta
    from src.database.models import PriceCache

    now = datetime.utcnow()
    for i in range(5):
        test_db.add(
            PriceCache(
                ticker="AAPL",
                date=datetime(2024, 1, 2 + i).date(),
                open=150.0 + i,
                high=155.0 + i,
                low=148.0 + i,
                close=152.0 + i,
                volume=1000000,
                created_at=now,
                updated_at=now,
            )
        )
    test_db.commit()

    response = api_client.get("/api/v1/cache/info?ticker=AAPL")
    assert response.status_code == 200
    caches = response.json()["caches"]
    assert len(caches) == 1
    assert caches[0]["ticker"] == "AAPL"
    assert caches[0]["records_count"] == 5
    assert caches[0]["is_valid"] is True


def test_model_info_not_found(api_client):
    """Model info sem modelo retorna 404."""
    response = api_client.get("/api/v1/model/info/XXXX")
    assert response.status_code == 404


def test_delete_cache(api_client, test_db):
    """Deletar cache de um ticker."""
    from datetime import datetime
    from src.database.models import PriceCache

    now = datetime.utcnow()
    test_db.add(
        PriceCache(
            ticker="AAPL",
            date=datetime(2024, 1, 2).date(),
            close=152.0,
            volume=1000000,
            created_at=now,
            updated_at=now,
        )
    )
    test_db.commit()

    response = api_client.delete("/api/v1/cache/AAPL")
    assert response.status_code == 200
    assert response.json()["deleted_records"] >= 1
