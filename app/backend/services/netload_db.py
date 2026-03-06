# app/backend/services/netload_db.py

from __future__ import annotations

import os
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)

# ---------- DB URL ----------
# Default: a local sqlite file next to backend (adjust if you prefer another path)
DEFAULT_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "netload.db")
DEFAULT_SQLITE_PATH = os.path.abspath(DEFAULT_SQLITE_PATH)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

# For SQLite, this flag is needed for FastAPI multi-threaded request handling
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


# ---------- Tables ----------
class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    feeder_id: Mapped[str] = mapped_column(String(32), default="F10", index=True)
    target_date: Mapped[str] = mapped_column(String(10), index=True)  # "YYYY-MM-DD"

    issue_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    model_name: Mapped[str] = mapped_column(String(128), default="Transformer")
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    points: Mapped[list["ForecastPoint"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("forecast_runs.id", ondelete="CASCADE"),
        index=True,
    )

    # Store step index and timestamp so both UI + analysis is easy
    step_idx: Mapped[int] = mapped_column(Integer)  # 0..95
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    yhat_mw: Mapped[float] = mapped_column(Float)

    run: Mapped["ForecastRun"] = relationship(back_populates="points")


# Helpful composite indexes (fast history queries later)
Index("ix_points_run_step", ForecastPoint.run_id, ForecastPoint.step_idx)


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency (use later in routes)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()