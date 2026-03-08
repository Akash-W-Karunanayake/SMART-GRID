"""
Application settings loaded from environment / .env file.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    DSS_MODEL_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MASTER_DSS: str = "Master.dss"

    # API
    APP_NAME: str = "Self-Healing Backend"
    APP_VERSION: str = "0.2.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8100
    DEBUG: bool = True
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # Logging
    LOG_LEVEL: str = "INFO"

    # ML
    TRAINED_MODEL_PATH: Optional[str] = None
    DEVICE: str = "cpu"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def master_dss_path(self) -> Path:
        return self.DSS_MODEL_DIR / self.MASTER_DSS


settings = Settings()
