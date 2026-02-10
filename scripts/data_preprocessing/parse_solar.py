"""
Parse NASA solar irradiance CSV data and interpolate to 15-min resolution.

Input: NASA/POWER hourly GHI data (Wh/m^2)
Output: data/processed/solar_15min.csv
Columns: Timestamp, GHI_Wh_m2, GHI_pu
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_SOLAR_FILE, SOLAR_15MIN, PROCESSED_DIR, NASA_HEADER_SKIP, STC_IRRADIANCE

logger = logging.getLogger(__name__)


def parse_solar() -> pd.DataFrame:
    """Parse NASA solar irradiance data and produce 15-min resolution output."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading solar data from: {RAW_SOLAR_FILE}")

    # Read NASA CSV, skip metadata header lines
    df = pd.read_csv(RAW_SOLAR_FILE, skiprows=NASA_HEADER_SKIP)

    # Clean column names (strip whitespace)
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

    # Find the GHI column (may be named ALLSKY_SFC_SW_DWN or similar)
    ghi_col = None
    for col in df.columns:
        if "ALLSKY" in col.upper() or "SFC_SW" in col.upper() or "GHI" in col.upper():
            ghi_col = col
            break

    if ghi_col is None:
        # Fallback: last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ghi_col = [c for c in numeric_cols if c not in ["YEAR", "MO", "DY", "HR"]][0]

    logger.info(f"  Using GHI column: {ghi_col}")

    # Replace missing values (-999) with NaN
    df[ghi_col] = df[ghi_col].replace(-999, np.nan)

    # Create hourly DataFrame
    hourly = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "GHI_Wh_m2": df[ghi_col].astype(float),
    })

    hourly = hourly.set_index("Timestamp").sort_index()

    # Interpolate missing values (linear)
    hourly["GHI_Wh_m2"] = hourly["GHI_Wh_m2"].interpolate(method="linear", limit=3)

    # Clamp negatives to 0
    hourly["GHI_Wh_m2"] = hourly["GHI_Wh_m2"].clip(lower=0)

    # Resample to 15-min intervals using linear interpolation
    fifteen_min = hourly.resample("15min").interpolate(method="linear")

    # Clamp negatives again after cubic interpolation (can introduce small negatives)
    fifteen_min["GHI_Wh_m2"] = fifteen_min["GHI_Wh_m2"].clip(lower=0)

    # Normalize to per-unit (GHI / STC_IRRADIANCE)
    fifteen_min["GHI_pu"] = fifteen_min["GHI_Wh_m2"] / STC_IRRADIANCE

    # Clamp pu to [0, 1.2] (allow slight over-irradiance)
    fifteen_min["GHI_pu"] = fifteen_min["GHI_pu"].clip(lower=0, upper=1.2)

    # Reset index
    result = fifteen_min.reset_index()

    # Save
    result.to_csv(SOLAR_15MIN, index=False)
    logger.info(f"Saved 15-min solar data: {SOLAR_15MIN}")
    logger.info(f"  Total rows: {len(result)}")
    logger.info(f"  Date range: {result['Timestamp'].min()} to {result['Timestamp'].max()}")
    logger.info(f"  GHI range: [{result['GHI_Wh_m2'].min():.1f}, {result['GHI_Wh_m2'].max():.1f}] Wh/m2")
    logger.info(f"  GHI_pu range: [{result['GHI_pu'].min():.3f}, {result['GHI_pu'].max():.3f}]")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_solar()
