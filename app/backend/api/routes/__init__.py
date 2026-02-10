"""API Routes package."""
from .grid import router as grid_router
from .simulation import router as simulation_router
from .forecasting import router as forecasting_router
from .diagnostics import router as diagnostics_router
from .pipeline import router as pipeline_router

__all__ = [
    "grid_router", "simulation_router", "forecasting_router",
    "diagnostics_router", "pipeline_router",
]
