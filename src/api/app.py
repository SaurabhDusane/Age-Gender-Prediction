from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app
import logging

from src.api.routes import analysis, streaming, faces, sessions, health
from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Demographic Analysis System API",
        description="Production-grade real-time facial attribute detection and tracking",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
    app.include_router(streaming.router, prefix="/api/v1", tags=["Streaming"])
    app.include_router(faces.router, prefix="/api/v1", tags=["Faces"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
    
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Demographic Analysis System API")
        from src.database.database import db
        db.create_tables()
        logger.info("Database tables created")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Demographic Analysis System API")
    
    return app


app = create_app()
