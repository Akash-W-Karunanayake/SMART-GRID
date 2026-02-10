"""
Load Disaggregation: decompose feeder-level net metered load into individual load profiles.

For each feeder and each 15-min interval on the target date:
1. Estimate PV generation from solar data
2. For F11: also estimate wind generation
3. Compute gross load = net load + PV generation [+ wind for F11]
4. Disaggregate gross load to individual loads using synthetic profile weighting
5. Compute per-unit multipliers for each load

Special case: F05 (UJPS) - no loads, just generator dispatch.

Output: CSV files in data/processed/disaggregated/ with per-load multiplier arrays.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOAD_PROFILES_CLEANED, SOLAR_15MIN, WIND_15MIN,
    DISAGGREGATED_DIR, FEEDER_LOADS, FEEDER_PV,
    SYNTHETIC_PROFILES, LOAD_FEEDERS,
    PV_SYSTEM_EFFICIENCY, MIN_GROSS_LOAD_FRACTION,
    WIND_FARM_CAPACITY_KW,
)

logger = logging.getLogger(__name__)


def _get_synthetic_weight(load_info: dict, hour: float) -> float:
    """Get the synthetic profile weight for a load at a given hour.

    Interpolates between hourly values for 15-min resolution.
    """
    profile_name = load_info["synthetic_profile"]
    profile = SYNTHETIC_PROFILES[profile_name]

    # Linear interpolation between hourly values
    hour_idx = int(hour) % 24
    next_idx = (hour_idx + 1) % 24
    frac = hour - int(hour)

    value = profile[hour_idx] * (1 - frac) + profile[next_idx] * frac
    return value


def _get_feeder_total_pv_kw(feeder: str) -> float:
    """Get total PV capacity for a feeder in kW."""
    if feeder not in FEEDER_PV:
        return 0.0
    return sum(pv["kw"] for pv in FEEDER_PV[feeder])


def _get_feeder_total_load_kw(feeder: str) -> float:
    """Get total rated load capacity for a feeder in kW."""
    if feeder not in FEEDER_LOADS:
        return 0.0
    return sum(load["kw"] for load in FEEDER_LOADS[feeder])


def disaggregate(target_date: str) -> dict:
    """Run disaggregation for a target date.

    Args:
        target_date: Date string in YYYY-MM-DD format (e.g. "2025-08-01")

    Returns:
        Dict mapping load/generator names to their 96-value multiplier arrays.
    """
    DISAGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.replace("-", "")  # "20250801"

    logger.info(f"Disaggregating loads for date: {target_date}")

    # Load preprocessed data
    load_df = pd.read_csv(LOAD_PROFILES_CLEANED, parse_dates=["Timestamp"])
    solar_df = pd.read_csv(SOLAR_15MIN, parse_dates=["Timestamp"])
    wind_df = pd.read_csv(WIND_15MIN, parse_dates=["Timestamp"])

    # Filter to target date
    target = pd.Timestamp(target_date)
    target_end = target + pd.Timedelta(days=1)

    solar_day = solar_df[
        (solar_df["Timestamp"] >= target) & (solar_df["Timestamp"] < target_end)
    ].reset_index(drop=True)

    wind_day = wind_df[
        (wind_df["Timestamp"] >= target) & (wind_df["Timestamp"] < target_end)
    ].reset_index(drop=True)

    if len(solar_day) == 0:
        raise ValueError(f"No solar data found for {target_date}")
    if len(wind_day) == 0:
        raise ValueError(f"No wind data found for {target_date}")

    # Ensure exactly 96 points (resample if needed)
    expected_ts = pd.date_range(start=target, periods=96, freq="15min")

    solar_day = solar_day.set_index("Timestamp").reindex(expected_ts).interpolate().reset_index()
    solar_day.columns = ["Timestamp"] + list(solar_day.columns[1:])

    wind_day = wind_day.set_index("Timestamp").reindex(expected_ts).interpolate().reset_index()
    wind_day.columns = ["Timestamp"] + list(wind_day.columns[1:])

    ghi_pu = solar_day["GHI_pu"].values[:96]
    wind_pu = wind_day["WindPower_pu"].values[:96]

    # Results dictionary: name -> 96 multiplier values
    all_multipliers = {}

    # ========================================================================
    # Process each load feeder (F06-F12)
    # ========================================================================
    for feeder in LOAD_FEEDERS:
        logger.info(f"  Processing {feeder}...")

        # Get feeder net load for target date
        feeder_data = load_df[
            (load_df["Feeder"] == feeder) &
            (load_df["Timestamp"] >= target) &
            (load_df["Timestamp"] < target_end)
        ].sort_values("Timestamp").reset_index(drop=True)

        if len(feeder_data) == 0:
            logger.warning(f"    No load data for {feeder} on {target_date}, using synthetic profiles")
            # Fall back to synthetic profiles
            for load_info in FEEDER_LOADS[feeder]:
                mults = []
                for i in range(96):
                    hour = i * 0.25
                    mults.append(_get_synthetic_weight(load_info, hour))
                all_multipliers[load_info["name"]] = mults
            continue

        # Resample/align to 96 intervals (drop duplicate timestamps first)
        feeder_data = feeder_data.drop_duplicates(subset="Timestamp", keep="last")
        feeder_data = feeder_data.set_index("Timestamp").sort_index()
        feeder_aligned = feeder_data["MW"].reindex(expected_ts, method="nearest", tolerance="30min")
        feeder_aligned = feeder_aligned.interpolate(method="linear").bfill().ffill()
        net_load_mw = feeder_aligned.values[:96]

        # Convert to kW
        net_load_kw = net_load_mw * 1000.0

        # Step 1: Estimate PV generation for this feeder
        total_pv_kw = _get_feeder_total_pv_kw(feeder)
        pv_gen_kw = total_pv_kw * ghi_pu * PV_SYSTEM_EFFICIENCY  # 96 values

        # Step 2: For F11, also estimate wind generation
        wind_gen_kw = np.zeros(96)
        if feeder == "F11":
            wind_gen_kw = WIND_FARM_CAPACITY_KW * wind_pu

        # Step 3: Estimate gross load
        # Net load = Gross load - PV - Wind (from substation perspective, net load is what's measured)
        # So: Gross load = Net load + PV + Wind
        gross_load_kw = net_load_kw + pv_gen_kw + wind_gen_kw

        # Clamp to minimum
        total_load_cap = _get_feeder_total_load_kw(feeder)
        min_load = total_load_cap * MIN_GROSS_LOAD_FRACTION
        gross_load_kw = np.maximum(gross_load_kw, min_load)

        logger.info(f"    Net load range: [{net_load_kw.min():.0f}, {net_load_kw.max():.0f}] kW")
        logger.info(f"    PV gen range: [{pv_gen_kw.min():.0f}, {pv_gen_kw.max():.0f}] kW")
        logger.info(f"    Gross load range: [{gross_load_kw.min():.0f}, {gross_load_kw.max():.0f}] kW")

        # Step 4: Disaggregate using synthetic profile weighting
        loads = FEEDER_LOADS[feeder]

        for i in range(96):
            hour = i * 0.25

            # Compute weights for all loads
            weights = []
            for load_info in loads:
                w = load_info["kw"] * _get_synthetic_weight(load_info, hour)
                weights.append(w)

            total_weight = sum(weights)
            if total_weight == 0:
                total_weight = 1.0  # Prevent division by zero

            # Allocate gross load proportionally
            for j, load_info in enumerate(loads):
                fraction = weights[j] / total_weight
                load_kw = gross_load_kw[i] * fraction

                # Step 5: Per-unit multiplier
                nominal_kw = load_info["kw"]
                mult = load_kw / nominal_kw if nominal_kw > 0 else 0.0

                # Clamp multiplier to reasonable range
                mult = max(0.01, min(mult, 3.0))

                name = load_info["name"]
                if name not in all_multipliers:
                    all_multipliers[name] = []
                all_multipliers[name].append(round(mult, 4))

    # ========================================================================
    # F05 (UJPS) - Generator dispatch
    # ========================================================================
    logger.info("  Processing F05 (UJPS)...")

    f05_data = load_df[
        (load_df["Feeder"] == "F05") &
        (load_df["Timestamp"] >= target) &
        (load_df["Timestamp"] < target_end)
    ].sort_values("Timestamp").reset_index(drop=True)

    if len(f05_data) > 0:
        f05_data = f05_data.drop_duplicates(subset="Timestamp", keep="last")
        f05_aligned = f05_data.set_index("Timestamp").sort_index()["MW"].reindex(
            expected_ts, method="nearest", tolerance="30min"
        ).interpolate(method="linear").bfill().ffill()
        ujps_mw = f05_aligned.values[:96]
        ujps_kw = np.abs(ujps_mw) * 1000.0  # Total UJPS output

        # Split equally among 3 generators, convert to per-unit of 8000 kW each
        ujps_per_gen_pu = (ujps_kw / 3.0) / 8000.0
        ujps_per_gen_pu = np.clip(ujps_per_gen_pu, 0.01, 1.0)
        ujps_mult = [round(v, 4) for v in ujps_per_gen_pu]
    else:
        logger.warning("    No F05 data, using synthetic Thermal_Daily profile")
        thermal = [
            0.75, 0.70, 0.65, 0.60, 0.62, 0.70, 0.75, 0.60,
            0.45, 0.30, 0.20, 0.15, 0.18, 0.22, 0.28, 0.40,
            0.55, 0.70, 0.85, 1.00, 0.95, 0.88, 0.82, 0.78,
        ]
        ujps_mult = []
        for i in range(96):
            hour = i * 0.25
            h = int(hour) % 24
            frac = hour - int(hour)
            val = thermal[h] * (1 - frac) + thermal[(h + 1) % 24] * frac
            ujps_mult.append(round(val, 4))

    all_multipliers["UJPS_Gen1"] = ujps_mult
    all_multipliers["UJPS_Gen2"] = ujps_mult
    all_multipliers["UJPS_Gen3"] = ujps_mult

    # ========================================================================
    # Solar shape (shared across all PV systems)
    # ========================================================================
    solar_mult = [round(v, 4) for v in ghi_pu]
    all_multipliers["Solar_Shape"] = solar_mult

    # ========================================================================
    # Wind shape
    # ========================================================================
    wind_mult = [round(v, 4) for v in wind_pu]
    all_multipliers["Wind_Shape"] = wind_mult

    # ========================================================================
    # Save all multipliers
    # ========================================================================
    for name, mults in all_multipliers.items():
        out_path = DISAGGREGATED_DIR / f"{name}_{date_str}.csv"
        pd.DataFrame({"multiplier": mults}).to_csv(out_path, index=False)

    logger.info(f"  Saved {len(all_multipliers)} multiplier files to {DISAGGREGATED_DIR}")

    return all_multipliers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    disaggregate("2025-08-01")
