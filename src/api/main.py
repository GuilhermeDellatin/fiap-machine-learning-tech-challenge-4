"""
FastAPI application.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database.connection import init_db
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.mlflow_setup import setup_mlflow

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    # Startup
    logger.info("Starting up...")
    init_db()
    logger.info("Database initialized")
    try:
        experiment_id = setup_mlflow()
        logger.info(
            "MLflow initialized "
            f"(experiment_id={experiment_id}, tracking_uri={settings.MLFLOW_TRACKING_URI})"
        )
    except Exception as e:
        logger.warning(
            "MLflow initialization failed at startup. "
            f"API will continue without MLflow until it becomes available: {e}"
        )

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Metrics middleware
from src.api.middleware.metrics import MetricsMiddleware

app.add_middleware(MetricsMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from src.api.routes import health, training, prediction, inference

app.include_router(health.router)
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])


@app.get("/")
def root():
    return {"name": "Stock Price Prediction API", "version": "1.0.0", "docs": "/docs"}
