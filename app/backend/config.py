"""
Application configuration settings.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Smart Grid AI Framework"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DSS_MODEL_DIR: Path = BASE_DIR  # OpenDSS files are in the root
    MASTER_DSS_FILE: str = "Master.dss"

    # Simulation settings
    SIMULATION_STEP_SECONDS: float = 1.0  # Time between simulation steps
    DEFAULT_SIMULATION_HOURS: int = 24  # Default simulation duration

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds

    # Database (future)
    DATABASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_dss_master_path() -> Path:
    """Get the full path to the Master.dss file."""
    return settings.DSS_MODEL_DIR / settings.MASTER_DSS_FILE


def get_dss_file_path(filename: str) -> Path:
    """Get the full path to any DSS file."""
    return settings.DSS_MODEL_DIR / filename
