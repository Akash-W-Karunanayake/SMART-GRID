import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IN_DIR = REPO_ROOT / "data" / "raw" / "netload"
OUT_DIR = REPO_ROOT / "data" / "processed" / "netload"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(IN_DIR.glob("F10_*.csv"))
if not files:
    raise FileNotFoundError(f"No F10_*.csv files found in: {IN_DIR}")

dfs = []
for f in files:
    df = pd.read_csv(f, header=3)
    df = df[["Timestamp", "MW"]].copy()

    # MW numeric
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")

    # robust timestamp parse for mixed formats
    ts = df["Timestamp"].astype(str).str.strip()
    ts_dash = pd.to_datetime(ts.where(ts.str.contains("-")), format="%d-%m-%Y %H:%M", errors="coerce")
    ts_slash = pd.to_datetime(ts.where(ts.str.contains("/")), dayfirst=True, errors="coerce")
    df["Timestamp"] = ts_dash.fillna(ts_slash)

    df = df.dropna(subset=["Timestamp", "MW"])
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged = merged.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")

# keep the original sheet MW as MW_raw
merged = merged.rename(columns={"MW": "MW_raw"})

# sign convention -> standard net load (Demand - Generation)
merged["NetLoad_MW"] = -merged["MW_raw"]

# keep Timestamp as a normal column (better for Excel + APIs)
merged = merged.sort_values("Timestamp")
merged["Timestamp"] = merged["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

out_path = OUT_DIR / "f10_netload_clean.csv"
merged.to_csv(out_path, index=False)

print("Merged files:", len(files))
print("Saved:", out_path)
print("Rows:", len(merged))
print("Start:", merged["Timestamp"].min(), "End:", merged["Timestamp"].max())
