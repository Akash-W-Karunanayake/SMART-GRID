import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# -------- Paths --------
F10_PATH  = REPO_ROOT / "data" / "processed" / "netload" / "f10_netload_clean.csv"
NASA_PATH = REPO_ROOT / "data" / "raw" / "netload" / "nasa_power_chunnakam_hourly.csv"

OUT_DIR  = REPO_ROOT / "data" / "processed" / "netload"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "f10_netload_nasa_clean.csv"

# -------- 1) Load F10 (15-min) --------
f10 = pd.read_csv(F10_PATH, parse_dates=["Timestamp"]).sort_values("Timestamp")

# -------- 2) Load NASA (hourly; skip header block) --------
# Your file has "-END HEADER-" on row 14 and column names on row 15 -> skiprows=14
nasa = pd.read_csv(NASA_PATH, skiprows=14)
nasa.columns = [c.strip() for c in nasa.columns]

# NASA missing code
nasa = nasa.replace([-999, -999.0], pd.NA)

# Build timestamp from YEAR/MO/DY/HR
nasa["Timestamp"] = pd.to_datetime(
    nasa["YEAR"].astype(int).astype(str) + "-" +
    nasa["MO"].astype(int).astype(str).str.zfill(2) + "-" +
    nasa["DY"].astype(int).astype(str).str.zfill(2) + " " +
    nasa["HR"].astype(int).astype(str).str.zfill(2) + ":00:00"
)

# Keep only necessary columns (research-safe set)
keep_cols = ["Timestamp", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M"]
keep_cols = [c for c in keep_cols if c in nasa.columns]
nasa = nasa[keep_cols].sort_values("Timestamp")

# Ensure numeric
for c in keep_cols:
    if c != "Timestamp":
        nasa[c] = pd.to_numeric(nasa[c], errors="coerce")

# -------- 3) Hourly -> 15-min (most accurate & safe under time) --------
nasa_15 = (
    nasa.set_index("Timestamp")
        .sort_index()
        .resample("15min")
        .ffill()
        .reset_index()
)

# -------- 4) Merge --------
merged = pd.merge(f10, nasa_15, on="Timestamp", how="left")

# Optional: force ISO timestamp string for Excel consistency
merged["Timestamp"] = merged["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

merged.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(merged))
print("Start:", merged["Timestamp"].min(), "End:", merged["Timestamp"].max())
print("NASA missing (ALLSKY):", merged["ALLSKY_SFC_SW_DWN"].isna().sum() if "ALLSKY_SFC_SW_DWN" in merged.columns else "n/a")