"""
Endpoints de health check e cache.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime

from src.api.dependencies import get_database, get_collector
from src.database.repository import PriceCacheRepository, ModelRegistryRepository
from src.api.schemas.prediction import CacheInfoResponse, CacheInfoItem, CacheSyncResponse
from src.monitoring.metrics import get_metrics
from src.utils.config import settings

router = APIRouter()


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type="text/plain"
    )


@router.get("/health")
def health_check(db: Session = Depends(get_database)):
    """Health check."""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e), "timestamp": datetime.utcnow()}


@router.get("/api/v1/model/info/{ticker}")
def get_model_info(ticker: str, db: Session = Depends(get_database)):
    """Info do modelo ativo para um ticker."""
    repo = ModelRegistryRepository()
    model = repo.get_active_model(db, ticker.upper())

    if not model:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No model found for ticker {ticker}",
                "suggestion": "Train a model using POST /api/v1/training/start",
            },
        )

    return {
        "ticker": model.ticker,
        "active_model": {
            "version_id": model.version_id,
            "mae": model.mae,
            "rmse": model.rmse,
            "mape": model.mape,
            "r2_score": model.r2_score,
            "epochs_trained": model.epochs_trained,
            "created_at": model.created_at,
        },
    }


@router.get("/api/v1/cache/info")
def get_cache_info(ticker: str = None, db: Session = Depends(get_database)):
    """Info do cache."""
    repo = PriceCacheRepository()

    if ticker:
        info = repo.get_cache_info(db, ticker.upper())
        if not info:
            return CacheInfoResponse(caches=[])

        last_updated = datetime.fromisoformat(info["last_updated"]) if info["last_updated"] else datetime.utcnow()
        age_hours = (datetime.utcnow() - last_updated).total_seconds() / 3600
        is_valid = age_hours < settings.CACHE_EXPIRY_HOURS
        expires_in = max(0, settings.CACHE_EXPIRY_HOURS - age_hours)

        return CacheInfoResponse(caches=[CacheInfoItem(
            ticker=ticker.upper(),
            records_count=info["record_count"],
            date_range=info["date_range"],
            last_updated=last_updated,
            is_valid=is_valid,
            expires_in_hours=round(expires_in, 2),
        )])

    return CacheInfoResponse(caches=[])


@router.post("/api/v1/cache/sync/{ticker}")
def sync_cache(ticker: str, db: Session = Depends(get_database)):
    """Força sincronização do cache."""
    collector = get_collector()
    df = collector.sync_data(db, ticker.upper())

    return CacheSyncResponse(
        ticker=ticker.upper(),
        records_updated=len(df),
        message="Cache synchronized successfully",
    )


@router.delete("/api/v1/cache/{ticker}")
def delete_cache(ticker: str, db: Session = Depends(get_database)):
    """Limpa cache de um ticker."""
    repo = PriceCacheRepository()

    if ticker.lower() == "all":
        deleted = repo.delete_cache(db)
    else:
        deleted = repo.delete_cache(db, ticker.upper())

    db.commit()
    return {"deleted_records": deleted, "ticker": ticker}
