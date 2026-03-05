from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from services.netload_forecaster import NetLoadForecaster

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

        return {
            "target_date": target_date,
            "issue_time": issue_time.isoformat(),
            "points": [{"timestamp": t.isoformat(), "yhat_mw": float(v)} for t, v in zip(ts, yhat)],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))