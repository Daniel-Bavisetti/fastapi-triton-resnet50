import logging

from fastapi import FastAPI

from app.core.config import get_settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Upjao Triton Inference API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(LoggingMiddleware)
app.include_router(health_router)
app.include_router(predict_router)
