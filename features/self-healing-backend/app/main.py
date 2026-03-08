"""
Self-Healing Backend — FastAPI Entry Point
Run: uvicorn app.main:app --reload --port 8100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from config import settings
from app.api.routes.isolation import router as isolation_router
from app.api.routes.restoration import router as restoration_router
from app.api.routes.simulation import router as simulation_router
from app.services.opendss_engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    ok = engine.load_model()
    if ok:
        logger.info("Chunnakam model loaded")
    else:
        logger.warning("Model not loaded — POST /api/v1/self-healing/load-model")
    logger.info(f"Docs: http://{settings.HOST}:{settings.PORT}/docs")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "## Self-Healing Grid Backend\n\n"
        "Fault isolation (rule-based) + MARL-based power restoration\n"
        "on the Chunnakam GSS OpenDSS model.\n\n"
        "**Component 4** — IT22053350\n\n"
        "Action space: **5 tie switches** (not all 21).\n"
        "Isolation switches (CBs + SECs) are rule-based."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1/self-healing"
app.include_router(simulation_router, prefix=PREFIX)
app.include_router(isolation_router, prefix=PREFIX)
app.include_router(restoration_router, prefix=PREFIX)


@app.get("/", tags=["Root"])
async def root():
    return {"service": settings.APP_NAME, "version": settings.APP_VERSION,
            "model_loaded": engine.is_loaded, "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": engine.is_loaded}
