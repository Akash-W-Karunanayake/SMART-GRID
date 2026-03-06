from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import select, delete
import logging

from services.netload_forecaster import NetLoadForecaster
from services.netload_db import SessionLocal, ForecastRun, ForecastPoint

logger = logging.getLogger(__name__)

#  Separate namespace so it won’t clash with solar/other forecasting routes
router = APIRouter(prefix="/netload", tags=["NetLoad Forecast"])

MODEL = NetLoadForecaster("artifacts_f10")


def issue_time_from_target_date(target_date: str) -> datetime:
    d = datetime.strptime(target_date, "%Y-%m-%d")
    return d - timedelta(minutes=15)   # prev day 23:45


def load_window(issue_time: datetime) -> np.ndarray:
    # TEMP: CSV read (replace with DB query later)
    df = pd.read_csv("data/processed/netload/f10_netload_nasa_clean.csv", parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp")

    end = issue_time
    start = end - timedelta(minutes=15 * (MODEL.L - 1))

    win = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
    if len(win) != MODEL.L:
        raise ValueError(f"Need {MODEL.L} rows, got {len(win)}.")

    cols = MODEL.feature_order  # ["NetLoad_MW","ALLSKY_SFC_SW_DWN","T2M","WS10M"]
    return win[cols].to_numpy(dtype=float)


def persist_forecast_to_db(
    feeder_id: str,
    target_date: str,
    issue_time: datetime,
    model_name: str,
    model_version: Optional[str],
    points: List[Dict[str, Any]],  # [{timestamp: datetime, yhat_mw: float, step_idx: int}]
) -> int:
    """
    Writes 1 run + 96 points to SQLite.
    Upsert-like behavior: overwrite points if run already exists for the same key.
    Returns run_id.
    """
    db = SessionLocal()
    try:
        # 1) Find existing run for same key (prevents duplicates in demo)
        stmt = select(ForecastRun).where(
            ForecastRun.feeder_id == feeder_id,
            ForecastRun.target_date == target_date,
            ForecastRun.issue_time == issue_time,
            ForecastRun.model_name == model_name,
        )
        existing = db.execute(stmt).scalar_one_or_none()

        if existing:
            run_id = existing.id
            # delete old points
            db.execute(delete(ForecastPoint).where(ForecastPoint.run_id == run_id))
        else:
            run = ForecastRun(
                feeder_id=feeder_id,
                target_date=target_date,
                issue_time=issue_time,
                model_name=model_name,
                model_version=model_version,
            )
            db.add(run)
            db.flush()  # gets run.id without commit
            run_id = run.id

        # 2) Insert 96 points
        db.add_all([
            ForecastPoint(
                run_id=run_id,
                step_idx=p["step_idx"],
                timestamp=p["timestamp"],
                yhat_mw=float(p["yhat_mw"]),
            )
            for p in points
        ])

        db.commit()
        return run_id

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@router.get("/forecast")
def forecast(target_date: str):
    """
    Day-ahead net load forecast for a target date.
    URL: /api/v1/netload/forecast?target_date=YYYY-MM-DD
    """
    try:
        issue_time = issue_time_from_target_date(target_date)
        X = load_window(issue_time)
        yhat = MODEL.predict(X)

        start_ts = datetime.strptime(target_date, "%Y-%m-%d")
        ts = [start_ts + timedelta(minutes=15 * i) for i in range(MODEL.H)]

        # --- Persist to DB (1 run + 96 points) ---
        run_id = None
        try:
            db_points = [
                {"step_idx": i, "timestamp": t, "yhat_mw": float(v)}
                for i, (t, v) in enumerate(zip(ts, yhat))
            ]

            run_id = persist_forecast_to_db(
                feeder_id="F10",
                target_date=target_date,
                issue_time=issue_time,
                model_name="TransformerNetLoad",
                model_version="best_transformer_cpu.pt",
                points=db_points,
            )
        except Exception as e:
            # Don’t break API if DB write fails
            logger.error(f"Failed to persist netload forecast to DB: {e}")

        return {
            "target_date": target_date,
            "issue_time": issue_time.isoformat(),
            # optional but useful for debugging/history later:
            "run_id": run_id,
            "points": [{"timestamp": t.isoformat(), "yhat_mw": float(v)} for t, v in zip(ts, yhat)],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))