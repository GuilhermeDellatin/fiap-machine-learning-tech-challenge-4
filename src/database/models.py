"""
Modelos SQLAlchemy para persistência.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    Boolean, BigInteger, Text, UniqueConstraint
)
from sqlalchemy.sql import func
from src.database.connection import Base


class PriceCache(Base):
    __tablename__ = "price_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    adj_close = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_ticker_date'),
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), unique=True, nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    status = Column(String(20), nullable=False, default='pending')
    epochs_total = Column(Integer, nullable=False)
    epochs_completed = Column(Integer, default=0)
    current_loss = Column(Float)
    best_loss = Column(Float)
    error_message = Column(Text)
    model_version_id = Column(String(50))
    hyperparameters = Column(Text)  # JSON string
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(String(50), unique=True, nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    model_path = Column(String(255), nullable=False)
    scaler_path = Column(String(255), nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    r2_score = Column(Float)
    epochs_trained = Column(Integer)
    hyperparameters = Column(Text)  # JSON string
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
