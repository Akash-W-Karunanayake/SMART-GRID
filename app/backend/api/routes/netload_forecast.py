from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import select, delete
from fastapi import Path
from sqlalchemy import func
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


def load_actual_points(target_date: str) -> List[Dict[str, Any]]:
    # TEMP: CSV read (replace with DB query later if needed)
    df = pd.read_csv("data/processed/netload/f10_netload_nasa_clean.csv", parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp")

    start_ts = datetime.strptime(target_date, "%Y-%m-%d")
    end_ts = start_ts + timedelta(days=1)

    day_df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].copy()

    if day_df.empty:
        return []

    # Only enable comparison when a full day exists
    if len(day_df) != MODEL.H:
        logger.warning(f"Actual data for {target_date} has {len(day_df)} rows, expected {MODEL.H}")
        return []

    return [
        {
            "timestamp": row["Timestamp"].isoformat(),
            "yhat_mw": float(row["NetLoad_MW"]),
        }
        for _, row in day_df.iterrows()
    ]


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


def _compute_threshold(points: list[float], mode: str, x: float) -> float:
    """
    mode:
      - "fixed": use x as MW threshold
      - "dynamic": X = x * daily_peak_abs (x is ratio, e.g., 0.05)
    """
    if not points:
        return float(x)

    if mode == "fixed":
        return float(x)

    # dynamic (robust): use abs peak so it works for negative netload too
    peak_abs = max(abs(v) for v in points)
    return float(x) * float(peak_abs)


def _label_and_action(yhat_mw: float, X: float) -> tuple[str, float, str]:
    """
    Returns: (label, severity, action)
    """
    if yhat_mw < -X:
        label = "oversupply"
        severity = abs(yhat_mw)  # MW magnitude
        action = "Charge battery."  # DSS-friendly, generic
    elif yhat_mw > X:
        label = "undersupply"
        severity = abs(yhat_mw)
        action = "Discharge battery."
    else:
        label = "balanced"
        severity = abs(yhat_mw)
        action = "No action."
    return label, severity, action 


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

        # Load actual net-load values for the same day (historical/test dates)
        actual_points = load_actual_points(target_date)

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

        computed_at = datetime.utcnow().isoformat() + "Z"

        return {
            "target_date": target_date,
            "issue_time": issue_time.isoformat(),
            "run_id": run_id,
            "computed_at": computed_at,
            "points": [
                {"timestamp": t.isoformat(), "yhat_mw": float(v)}
                for t, v in zip(ts, yhat)
            ],
            "actual_points": actual_points,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/history")
def history(limit: int = 10):
    """
    Returns latest forecast runs (audit trail).
    URL: /api/v1/netload/history?limit=10
    """
    db = SessionLocal()
    try:
        limit = max(1, min(int(limit), 200))  # safety clamp

        stmt = (
            select(ForecastRun)
            .order_by(ForecastRun.created_at.desc())
            .limit(limit)
        )
        runs = db.execute(stmt).scalars().all()

        return {
            "count": len(runs),
            "runs": [
                {
                    "run_id": r.id,
                    "feeder_id": r.feeder_id,
                    "target_date": r.target_date,
                    "issue_time": r.issue_time.isoformat(),
                    "model_name": r.model_name,
                    "model_version": r.model_version,
                    "created_at": r.created_at.isoformat(),
                }
                for r in runs
            ],
        }
    finally:
        db.close()

@router.get("/run/{run_id}")
def get_run(run_id: int):
    """
    Returns a specific forecast run + its 96 points.
    URL: /api/v1/netload/run/{run_id}
    """
    db = SessionLocal()
    try:
        run = db.get(ForecastRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        pts_stmt = (
            select(ForecastPoint)
            .where(ForecastPoint.run_id == run_id)
            .order_by(ForecastPoint.step_idx.asc())
        )
        points = db.execute(pts_stmt).scalars().all()

        return {
            "run_id": run.id,
            "feeder_id": run.feeder_id,
            "target_date": run.target_date,
            "issue_time": run.issue_time.isoformat(),
            "model_name": run.model_name,
            "model_version": run.model_version,
            "created_at": run.created_at.isoformat(),
            "points": [
                {
                    "step_idx": p.step_idx,
                    "timestamp": p.timestamp.isoformat(),
                    "yhat_mw": float(p.yhat_mw),
                }
                for p in points
            ],
        }
    finally:
        db.close()

@router.get("/run/{run_id}/imbalance")
def run_imbalance(
    run_id: int,
    threshold_mode: str = "dynamic",  # "dynamic" or "fixed"
    x: float = 0.20,                  # dynamic ratio OR fixed MW depending on mode
    top_k: int = 10
):
    """
    Classify each 15-min step as oversupply/balanced/undersupply + recommend action.
    URL:
      /api/v1/netload/run/{run_id}/imbalance?threshold_mode=dynamic&x=0.05&top_k=10
      /api/v1/netload/run/{run_id}/imbalance?threshold_mode=fixed&x=1.0&top_k=10
    """
    mode = (threshold_mode or "").lower().strip()
    if mode not in {"dynamic", "fixed"}:
        raise HTTPException(status_code=400, detail="threshold_mode must be 'dynamic' or 'fixed'")

    top_k = max(1, min(int(top_k), 50))

    db = SessionLocal()
    try:
        run = db.get(ForecastRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        pts_stmt = (
            select(ForecastPoint)
            .where(ForecastPoint.run_id == run_id)
            .order_by(ForecastPoint.step_idx.asc())
        )
        pts = db.execute(pts_stmt).scalars().all()
        if not pts:
            raise HTTPException(status_code=404, detail=f"No points found for run {run_id}")

        yvals = [float(p.yhat_mw) for p in pts]
        X = _compute_threshold(yvals, mode=mode, x=float(x))

        records = []
        counts = {"oversupply": 0, "balanced": 0, "undersupply": 0}

        for p in pts:
            label, severity, action = _label_and_action(float(p.yhat_mw), X)
            counts[label] += 1
            records.append({
                "step_idx": p.step_idx,
                "timestamp": p.timestamp.isoformat(),
                "yhat_mw": float(p.yhat_mw),
                "label": label,
                "severity": float(severity),
                "recommended_action": action,
            })

        # “Worst” intervals:
        worst_undersupply = sorted(
            [r for r in records if r["label"] == "undersupply"],
            key=lambda r: r["yhat_mw"],
            reverse=True
        )[:top_k]

        worst_oversupply = sorted(
            [r for r in records if r["label"] == "oversupply"],
            key=lambda r: r["yhat_mw"]
        )[:top_k]

        return {
            "run_id": run.id,
            "feeder_id": run.feeder_id,
            "target_date": run.target_date,
            "issue_time": run.issue_time.isoformat(),
            "model_name": run.model_name,
            "model_version": run.model_version,
            "threshold_mode": mode,
            "threshold_X_mw": float(X),
            "counts": counts,
            "top_k": top_k,
            "worst_undersupply": worst_undersupply,
            "worst_oversupply": worst_oversupply,
            # optional: full labeled timeline
            "timeline": records,
            "note": "Actions assume battery storage is available."
        }
    finally:
        db.close()