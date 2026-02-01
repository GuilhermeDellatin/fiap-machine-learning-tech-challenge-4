"""
FastAPI application.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database.connection import init_db
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    # Startup
    logger.info("Starting up...")
    init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction",
    version="1.0.0",
    lifespan=lifespan,
)

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
