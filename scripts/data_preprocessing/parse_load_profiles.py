"""
Parse CEB metering load profile XLS files from data/raw/load_profiles/.
Handles mixed .xls/.XLS extensions and various data formats.

Output: data/processed/load_profiles_cleaned.csv
Columns: Timestamp, Feeder, MW, Mvar, MVA, is_export
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_LOAD_PROFILES_DIR, LOAD_PROFILES_CLEANED, PROCESSED_DIR, ALL_FEEDERS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Month name mapping for filename parsing
MONTH_MAP = {
    "may": 5, "june": 6, "july": 7, "august": 8,
    "april": 4, "jan": 1, "feb": 2, "mar": 3,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_filename(filepath: Path):
    """Extract feeder and month from filename like F06_August.xls."""
    stem = filepath.stem  # e.g. "F06_August"
    parts = stem.split("_", 1)
    feeder = parts[0].upper()
    month_str = parts[1].lower() if len(parts) > 1 else ""
    month = MONTH_MAP.get(month_str)
    return feeder, month


def _read_xls_file(filepath: Path) -> pd.DataFrame:
    """Read an XLS file, trying multiple strategies."""
    # Try standard Excel read first
    try:
        df = pd.read_excel(filepath, engine="xlrd")
        if len(df) > 0:
            return df
    except Exception:
        pass

    # Fallback: try as tab-separated CSV
    try:
        df = pd.read_csv(filepath, sep="\t")
        if len(df) > 0:
            return df
    except Exception:
        pass

    # Fallback: try comma-separated CSV
    try:
        df = pd.read_csv(filepath, sep=",")
        if len(df) > 0:
            return df
    except Exception:
        pass

    raise ValueError(f"Could not read file: {filepath}")


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, handling various formats."""
    return pd.to_numeric(series, errors="coerce")


def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """Try to find and parse timestamp column.

    Common formats: DD-MM-YYYY HH:MM, DD/MM/YYYY HH:MM, or separate Date/Time columns.
    """
    # Check for a combined datetime column
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(kw in col_lower for kw in ["date", "time", "timestamp"]):
            try:
                # Try DD-MM-YYYY HH:MM format first (common in CEB data)
                ts = pd.to_datetime(df[col], format="%d-%m-%Y %H:%M", errors="coerce")
                if ts.notna().sum() > len(df) * 0.5:
                    return ts
            except Exception:
                pass
            try:
                ts = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                if ts.notna().sum() > len(df) * 0.5:
                    return ts
            except Exception:
                pass

    # Try first column as timestamp
    try:
        ts = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
        if ts.notna().sum() > len(df) * 0.5:
            return ts
    except Exception:
        pass

    return None


def _find_power_columns(df: pd.DataFrame):
    """Find MW, Mvar, MVA columns by name patterns."""
    mw_col = mvar_col = mva_col = None
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "mw" in col_lower and "mvar" not in col_lower and "mva" not in col_lower:
            mw_col = col
        elif "mvar" in col_lower:
            mvar_col = col
        elif "mva" in col_lower:
            mva_col = col

    # If no named columns found, use positional (assume cols after timestamp)
    # Typical layout: Timestamp, MW, Mvar, MVA
    if mw_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 1:
            mw_col = numeric_cols[0]
        if len(numeric_cols) >= 2:
            mvar_col = numeric_cols[1]
        if len(numeric_cols) >= 3:
            mva_col = numeric_cols[2]

    return mw_col, mvar_col, mva_col


def parse_single_file(filepath: Path, feeder: str, month: int) -> pd.DataFrame:
    """Parse a single load profile file into a cleaned DataFrame."""
    logger.info(f"Parsing {filepath.name} (Feeder={feeder}, Month={month})")

    df = _read_xls_file(filepath)

    if df.empty:
        logger.warning(f"  Empty file: {filepath.name}")
        return pd.DataFrame()

    # Parse timestamps
    timestamps = _parse_timestamp(df)

    # Find power columns
    mw_col, mvar_col, mva_col = _find_power_columns(df)

    if mw_col is None:
        logger.warning(f"  No MW column found in {filepath.name}")
        return pd.DataFrame()

    # Build clean DataFrame
    result = pd.DataFrame()

    if timestamps is not None:
        result["Timestamp"] = timestamps
    else:
        # Generate synthetic timestamps if none found
        # Assume 15-min intervals starting from first day of month
        year = 2025
        start = pd.Timestamp(year=year, month=month, day=1)
        result["Timestamp"] = pd.date_range(
            start=start, periods=len(df), freq="15min"
        )

    result["Feeder"] = feeder
    result["MW"] = _clean_numeric(df[mw_col])

    if mvar_col is not None:
        result["Mvar"] = _clean_numeric(df[mvar_col])
    else:
        # Estimate reactive power from MW assuming pf=0.9
        result["Mvar"] = result["MW"] * 0.484  # tan(acos(0.9))

    if mva_col is not None:
        result["MVA"] = _clean_numeric(df[mva_col])
    else:
        result["MVA"] = np.sqrt(result["MW"] ** 2 + result["Mvar"] ** 2)

    # Determine export flag (negative MW = export from substation perspective)
    # In CEB data: negative net load means power flowing from customers to grid
    result["is_export"] = result["MW"] < 0

    # Drop rows with NaN timestamps or MW
    result = result.dropna(subset=["Timestamp", "MW"])

    logger.info(f"  Parsed {len(result)} rows, MW range: [{result['MW'].min():.3f}, {result['MW'].max():.3f}]")

    return result


def parse_all_load_profiles() -> pd.DataFrame:
    """Parse all load profile files and combine into single DataFrame."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(RAW_LOAD_PROFILES_DIR.glob("*.[xX][lL][sS]"))
    logger.info(f"Found {len(all_files)} load profile files")

    frames = []
    for filepath in all_files:
        feeder, month = _parse_filename(filepath)
        if feeder not in ALL_FEEDERS:
            logger.warning(f"  Skipping unknown feeder: {feeder} in {filepath.name}")
            continue
        if month is None:
            logger.warning(f"  Could not determine month for {filepath.name}")
            continue

        try:
            df = parse_single_file(filepath, feeder, month)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.error(f"  Error parsing {filepath.name}: {e}")
            continue

    if not frames:
        raise RuntimeError("No load profile data could be parsed!")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Feeder", "Timestamp"]).reset_index(drop=True)

    # Interpolate small gaps (up to 4 intervals = 1 hour)
    for feeder in combined["Feeder"].unique():
        mask = combined["Feeder"] == feeder
        for col in ["MW", "Mvar", "MVA"]:
            combined.loc[mask, col] = (
                combined.loc[mask, col]
                .interpolate(method="linear", limit=4)
            )

    # Save
    combined.to_csv(LOAD_PROFILES_CLEANED, index=False)
    logger.info(f"Saved cleaned load profiles: {LOAD_PROFILES_CLEANED}")
    logger.info(f"  Total rows: {len(combined)}")
    logger.info(f"  Feeders: {sorted(combined['Feeder'].unique())}")
    logger.info(f"  Date range: {combined['Timestamp'].min()} to {combined['Timestamp'].max()}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_all_load_profiles()
