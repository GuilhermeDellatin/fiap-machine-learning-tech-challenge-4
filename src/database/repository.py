"""
Repositórios CRUD para as tabelas do banco de dados.
"""
import json
from datetime import date, datetime, timedelta
from typing import List

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.models import PriceCache, TrainingJob, ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceCacheRepository:
    """Repositório para cache de preços do yfinance."""

    def get_prices(
        self, db: Session, ticker: str, start_date: date, end_date: date
    ) -> List[PriceCache]:
        """Retorna registros do cache para o período."""
        return (
            db.query(PriceCache)
            .filter(
                PriceCache.ticker == ticker,
                PriceCache.date >= start_date,
                PriceCache.date <= end_date,
            )
            .order_by(PriceCache.date)
            .all()
        )

    def save_prices(self, db: Session, ticker: str, df: pd.DataFrame) -> int:
        """Salva DataFrame no cache. Retorna quantidade salva."""
        count = 0
        for idx, row in df.iterrows():
            if "Date" in row and row["Date"] is not None:
                raw = row["Date"]
                row_date = raw.date() if hasattr(raw, "date") and callable(raw.date) else raw
            elif hasattr(idx, "date") and callable(idx.date):
                row_date = idx.date()
            else:
                row_date = idx

            existing = (
                db.query(PriceCache)
                .filter(PriceCache.ticker == ticker, PriceCache.date == row_date)
                .first()
            )

            if existing:
                existing.open = float(row.get("Open", 0))
                existing.high = float(row.get("High", 0))
                existing.low = float(row.get("Low", 0))
                existing.close = float(row["Close"])
                existing.adj_close = float(row.get("Adj Close", row["Close"]))
                existing.volume = int(row.get("Volume", 0))
                existing.updated_at = datetime.utcnow()
            else:
                record = PriceCache(
                    ticker=ticker,
                    date=row_date,
                    open=float(row.get("Open", 0)),
                    high=float(row.get("High", 0)),
                    low=float(row.get("Low", 0)),
                    close=float(row["Close"]),
                    adj_close=float(row.get("Adj Close", row["Close"])),
                    volume=int(row.get("Volume", 0)),
                )
                db.add(record)
                count += 1

        db.commit()
        logger.info(f"Saved {count} new price records for {ticker}")
        return count

    def is_cache_valid(
        self,
        db: Session,
        ticker: str,
        start_date: date,
        end_date: date,
        max_age_hours: int = 24,
    ) -> bool:
        """
        REGRA DE EXPIRAÇÃO:
        1. Verificar se existem dados no período
        2. Verificar se updated_at do registro mais recente < max_age_hours

        Retorna True se cache válido, False se expirado/inexistente.
        """
        latest = (
            db.query(PriceCache)
            .filter(
                PriceCache.ticker == ticker,
                PriceCache.date >= start_date,
                PriceCache.date <= end_date,
            )
            .order_by(PriceCache.updated_at.desc())
            .first()
        )

        if not latest or not latest.updated_at:
            return False

        age = datetime.utcnow() - latest.updated_at
        return age < timedelta(hours=max_age_hours)

    def get_cache_info(self, db: Session, ticker: str) -> dict | None:
        """Retorna info do cache: record_count, date_range, last_updated."""
        records = (
            db.query(PriceCache).filter(PriceCache.ticker == ticker).all()
        )

        if not records:
            return None

        dates = [r.date for r in records]
        updated_times = [r.updated_at for r in records if r.updated_at]

        return {
            "record_count": len(records),
            "date_range": {
                "start": min(dates).isoformat(),
                "end": max(dates).isoformat(),
            },
            "last_updated": max(updated_times).isoformat() if updated_times else None,
        }

    def delete_cache(self, db: Session, ticker: str = None) -> int:
        """Limpa cache. Se ticker=None, limpa tudo."""
        query = db.query(PriceCache)
        if ticker:
            query = query.filter(PriceCache.ticker == ticker)

        count = query.count()
        query.delete()
        db.commit()
        logger.info(f"Deleted {count} cache records" + (f" for {ticker}" if ticker else ""))
        return count


class TrainingJobRepository:
    """Repositório para jobs de treinamento."""

    def create_job(
        self, db: Session, job_id: str, ticker: str, epochs: int, hyperparams: dict
    ) -> TrainingJob:
        """Cria job com status='pending'."""
        job = TrainingJob(
            job_id=job_id,
            ticker=ticker,
            status="pending",
            epochs_total=epochs,
            hyperparameters=json.dumps(hyperparams),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info(f"Created training job {job_id} for {ticker}")
        return job

    def update_job(self, db: Session, job_id: str, **kwargs) -> TrainingJob | None:
        """Atualiza campos do job."""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return None

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        db.commit()
        db.refresh(job)
        return job

    def get_job(self, db: Session, job_id: str) -> TrainingJob | None:
        """Retorna job pelo job_id."""
        return db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    def list_jobs(
        self, db: Session, ticker: str = None, status: str = None, limit: int = 50
    ) -> List[TrainingJob]:
        """Lista jobs com filtros opcionais."""
        query = db.query(TrainingJob)

        if ticker:
            query = query.filter(TrainingJob.ticker == ticker)
        if status:
            query = query.filter(TrainingJob.status == status)

        return query.order_by(TrainingJob.created_at.desc()).limit(limit).all()


class ModelRegistryRepository:
    """Repositório para registro de modelos treinados."""

    def register_model(
        self,
        db: Session,
        ticker: str,
        model_path: str,
        scaler_path: str,
        metrics: dict,
        hyperparams: dict,
        epochs: int,
    ) -> ModelRegistry:
        """
        Registra modelo. Gera version_id automaticamente.
        Formato: {ticker}_{timestamp}
        """
        version_id = f"{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        model = ModelRegistry(
            version_id=version_id,
            ticker=ticker,
            model_path=model_path,
            scaler_path=scaler_path,
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            mape=metrics.get("mape"),
            r2_score=metrics.get("r2_score"),
            epochs_trained=epochs,
            hyperparameters=json.dumps(hyperparams),
            is_active=False,
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        logger.info(f"Registered model {version_id} for {ticker}")
        return model

    def get_model(self, db: Session, version_id: str) -> ModelRegistry | None:
        """Retorna modelo pelo version_id."""
        return (
            db.query(ModelRegistry)
            .filter(ModelRegistry.version_id == version_id)
            .first()
        )

    def get_active_model(self, db: Session, ticker: str) -> ModelRegistry | None:
        """Retorna modelo com is_active=True para o ticker."""
        return (
            db.query(ModelRegistry)
            .filter(ModelRegistry.ticker == ticker, ModelRegistry.is_active == True)
            .first()
        )

    def set_active_model(self, db: Session, ticker: str, version_id: str) -> bool:
        """
        1. Desativa todos os modelos do ticker (is_active=False)
        2. Ativa o modelo especificado (is_active=True)
        """
        db.query(ModelRegistry).filter(ModelRegistry.ticker == ticker).update(
            {"is_active": False}
        )

        model = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.version_id == version_id)
            .first()
        )

        if not model:
            db.rollback()
            return False

        model.is_active = True
        db.commit()
        logger.info(f"Set active model for {ticker}: {version_id}")
        return True

    def list_models(
        self, db: Session, ticker: str = None, active_only: bool = False
    ) -> List[ModelRegistry]:
        """Lista modelos com filtros opcionais."""
        query = db.query(ModelRegistry)

        if ticker:
            query = query.filter(ModelRegistry.ticker == ticker)
        if active_only:
            query = query.filter(ModelRegistry.is_active == True)

        return query.order_by(ModelRegistry.created_at.desc()).all()
