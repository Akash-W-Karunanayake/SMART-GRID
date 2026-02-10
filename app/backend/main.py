"""
Smart Grid AI Framework - Backend API Server

Main entry point for the FastAPI application.
Provides REST API and WebSocket endpoints for power system simulation and AI integration.

Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from api.routes import grid_router, simulation_router, forecasting_router, diagnostics_router, pipeline_router
from api.websockets import websocket_endpoint, manager
from services import opendss_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Try to load the OpenDSS model on startup
    try:
        result = opendss_service.load_model()
        if result["success"]:
            logger.info(f"OpenDSS model loaded: {result['circuit_name']}")
        else:
            logger.warning(f"Failed to load OpenDSS model: {result.get('error')}")
    except Exception as e:
        logger.error(f"Error loading OpenDSS model: {e}")

    logger.info(f"API documentation available at: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"WebSocket endpoint: ws://{settings.HOST}:{settings.PORT}/ws")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down application...")
    logger.info("Application stopped.")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Smart Grid AI Framework API

    A comprehensive backend for power system simulation and AI-driven grid management.

    ### Features

    * **Grid Management**: Load and control OpenDSS power system models
    * **Real-time Simulation**: Run time-series simulations with WebSocket updates
    * **Forecasting**: Load, solar, and net-load forecasting (ML integration ready)
    * **Diagnostics**: Fault detection and self-healing capabilities (ML integration ready)

    ### Research Components

    This API supports four research sub-components:

    1. **Self-Healing Grid** (MARL + GNN) - Autonomous fault isolation and service restoration
    2. **Solar Forecasting** (Stacked Ensemble) - Weather-based solar generation prediction
    3. **Fault Diagnostics** (CNN-Transformer + R-GNN) - Fault detection, classification, location
    4. **Net Load Forecasting** (ICEEMDAN + Transformer + GP-RML) - Probabilistic forecasting

    ### WebSocket

    Connect to `/ws` for real-time simulation updates.
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(grid_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(forecasting_router, prefix="/api/v1")
app.include_router(diagnostics_router, prefix="/api/v1")
app.include_router(pipeline_router, prefix="/api/v1")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """
    WebSocket endpoint for real-time simulation data.

    Connect to receive live grid state updates during simulation.

    **Message Format (Client -> Server):**
    ```json
    {
        "action": "start|stop|pause|resume|step|set_speed|get_state|get_status|ping",
        "params": {}
    }
    ```

    **Message Format (Server -> Client):**
    ```json
    {
        "type": "state_update|status|error|info|pong",
        "data": {},
        "timestamp": "ISO timestamp"
    }
    ```
    """
    await websocket_endpoint(websocket)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Smart Grid AI Framework API",
        "docs": "/docs",
        "websocket": "/ws",
        "model_loaded": opendss_service.is_loaded
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": opendss_service.is_loaded,
        "websocket_connections": manager.connection_count
    }


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
