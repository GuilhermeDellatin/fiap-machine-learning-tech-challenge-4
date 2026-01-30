"""
Configurações centralizadas usando Pydantic Settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./data/stock_cache.db"
    CACHE_EXPIRY_HOURS: int = 24

    # Paths
    MODEL_DIR: str = "../models"
    DATA_DIR: str = "../data"

    # Model Architecture
    SEQUENCE_LENGTH: int = 60
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    INPUT_SIZE: int = 1

    # Training
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
