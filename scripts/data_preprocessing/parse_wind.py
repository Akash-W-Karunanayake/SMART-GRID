"""
Parse NASA wind speed CSV data, interpolate to 15-min, convert to power.

Input: NASA/POWER hourly wind speed at 50m (m/s)
Output: data/processed/wind_15min.csv
Columns: Timestamp, WindSpeed_ms, WindPower_kW, WindPower_pu
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_WIND_FILE, WIND_15MIN, PROCESSED_DIR, NASA_HEADER_SKIP,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT, WIND_FARM_CAPACITY_KW,
)

logger = logging.getLogger(__name__)


def wind_speed_to_power(speed: float) -> float:
    """Convert wind speed (m/s) to power output (kW) using turbine power curve.

    Piecewise linear model:
      - Below cut-in: 0
      - Cut-in to rated: cubic ramp (proportional to v^3)
      - Rated to cut-out: rated power
      - Above cut-out: 0
    """
    if speed < WIND_CUT_IN or speed > WIND_CUT_OUT:
        return 0.0
    elif speed <= WIND_RATED:
        # Cubic power curve between cut-in and rated
        fraction = ((speed - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** 3
        return WIND_FARM_CAPACITY_KW * fraction
    else:
        return WIND_FARM_CAPACITY_KW


def parse_wind() -> pd.DataFrame:
    """Parse NASA wind data and produce 15-min resolution output with power conversion."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading wind data from: {RAW_WIND_FILE}")

    # Read NASA CSV, skip metadata header lines
    df = pd.read_csv(RAW_WIND_FILE, skiprows=NASA_HEADER_SKIP)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    logger.info(f"  Columns found: {list(df.columns)}")
    logger.info(f"  Raw rows: {len(df)}")

    # Build datetime from YEAR, MO, DY, HR columns
    df["Timestamp"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" +
        df["MO"].astype(str).str.zfill(2) + "-" +
        df["DY"].astype(str).str.zfill(2) + " " +
        df["HR"].astype(str).str.zfill(2) + ":00:00"
    )

    # Find wind speed column
    ws_col = None
    for col in df.columns:
        if "WS50M" in col.upper() or "WIND" in col.upper():
            ws_col = col
            break

    if ws_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ws_col = [c for c in numeric_cols if c not in ["YEAR", "MO", "DY", "HR"]][0]

    logger.info(f"  Using wind speed column: {ws_col}")

    # Replace missing values (-999) with NaN
    df[ws_col] = df[ws_col].replace(-999, np.nan)

    # Create hourly DataFrame
    hourly = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "WindSpeed_ms": df[ws_col].astype(float),
    })

    hourly = hourly.set_index("Timestamp").sort_index()

    # Interpolate missing values
    hourly["WindSpeed_ms"] = hourly["WindSpeed_ms"].interpolate(method="linear", limit=3)

    # Apply 3-hour rolling average to smooth timing mismatches between
    # NASA hourly data and actual wind output (reduces negative correlation days)
    hourly["WindSpeed_ms"] = hourly["WindSpeed_ms"].rolling(3, center=True, min_periods=1).mean()

    # Clamp negatives to 0
    hourly["WindSpeed_ms"] = hourly["WindSpeed_ms"].clip(lower=0)

    # Resample to 15-min intervals (linear interpolation for wind)
    fifteen_min = hourly.resample("15min").interpolate(method="linear")

    # Convert wind speed to power
    fifteen_min["WindPower_kW"] = fifteen_min["WindSpeed_ms"].apply(wind_speed_to_power)

    # Per-unit based on farm capacity
    fifteen_min["WindPower_pu"] = fifteen_min["WindPower_kW"] / WIND_FARM_CAPACITY_KW

    # Reset index
    result = fifteen_min.reset_index()

    # Save
    result.to_csv(WIND_15MIN, index=False)
    logger.info(f"Saved 15-min wind data: {WIND_15MIN}")
    logger.info(f"  Total rows: {len(result)}")
    logger.info(f"  Date range: {result['Timestamp'].min()} to {result['Timestamp'].max()}")
    logger.info(f"  Wind speed range: [{result['WindSpeed_ms'].min():.2f}, {result['WindSpeed_ms'].max():.2f}] m/s")
    logger.info(f"  Wind power range: [{result['WindPower_kW'].min():.1f}, {result['WindPower_kW'].max():.1f}] kW")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_wind()
