"""
Validation script: compare simulated feeder power against measured CEB data.

Compiles modified Master.dss, runs 96-step daily simulation, extracts
per-feeder power at feeder head lines, and compares against measured net load.

Metrics:
- MAE (target < 1.0 MW per feeder, < 5 MW system)
- MAPE (target < 15%)
- Correlation (target > 0.85)
- Voltage violations within 0.95-1.05 pu
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LOAD_PROFILES_CLEANED, MASTER_DSS, LOAD_FEEDERS,
)

logger = logging.getLogger(__name__)

# Feeder head line elements: first line section of each feeder after the CB.
# Power at bus1 (sending end) = net power flowing into the feeder from 33kV bus.
FEEDER_HEAD_LINES = {
    "F06": "Line.F06_Sec2",
    "F07": "Line.F07_Sec2",
    "F08": "Line.F08_Sec2",
    "F09": "Line.F09_Sec2",
    "F10": "Line.F10_Sec2",
    "F11": "Line.F11_Sec2",
    "F12": "Line.F12_Sec2",
}


def _robust_solve(dss):
    """Solve current timestep with generous settings.

    CRITICAL: Only call Solve() ONCE per step in daily mode.
    Each Solve() advances the simulation clock by one timestep.
    Calling Solve() multiple times = skipping timesteps and
    destroying the time alignment between simulated and measured data!
    """
    dss.Text.Command("Set algorithm=newton")
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Set tolerance=0.001")
    dss.Solution.Solve()
    return dss.Solution.Converged()


def _extract_feeder_power(dss, feeder, element_name):
    """Extract 3-phase real power at feeder head (sending end).

    Returns power in kW. Positive = load consuming, Negative = PV exporting.
    """
    try:
        dss.Circuit.SetActiveElement(element_name)
        powers = dss.CktElement.Powers()
        if powers and len(powers) >= 6:
            # Sum P at terminal 1 (sending end) for 3 phases
            # Powers format: [P1_t1, Q1_t1, P2_t1, Q2_t1, P3_t1, Q3_t1, ...]
            return powers[0] + powers[2] + powers[4]
    except Exception:
        pass
    return np.nan


def _try_opendss_simulation():
    """Attempt to run OpenDSS daily simulation and return results.

    Returns:
        List of dicts with per-step data, or None if OpenDSS not available.
    """
    try:
        import opendssdirect as dss
    except ImportError:
        logger.warning("opendssdirect not installed - skipping simulation validation")
        return None

    logger.info(f"Compiling: {MASTER_DSS}")
    dss.Basic.ClearAll()
    dss.Basic.DataPath(str(MASTER_DSS.parent))
    dss.Text.Command(f'Compile "{MASTER_DSS}"')

    error = dss.Error.Description()
    if error:
        logger.error(f"Compile error: {error}")
        return None

    circuit_name = dss.Circuit.Name()
    if not circuit_name:
        logger.error("No circuit loaded after compile")
        return None

    logger.info(f"Circuit loaded: {circuit_name}")

    # Run daily simulation
    dss.Text.Command("Set Mode=Daily")
    dss.Text.Command("Set Stepsize=15m")
    dss.Text.Command("Set Number=1")
    dss.Text.Command("Set controlmode=static")

    results = []
    for step in range(96):
        converged = _robust_solve(dss)

        step_data = {
            "step": step,
            "hour": step * 0.25,
            "converged": converged,
        }

        if converged:
            # Source power (for reference)
            step_data["source_power_kw"] = -dss.Circuit.TotalPower()[0]

            # Per-feeder power at feeder head lines
            for feeder, element in FEEDER_HEAD_LINES.items():
                step_data[f"power_{feeder}_kw"] = _extract_feeder_power(
                    dss, feeder, element
                )

            # Bus voltages for violation checking
            bus_names = dss.Circuit.AllBusNames()
            voltages = []
            for bname in bus_names:
                dss.Circuit.SetActiveBus(bname)
                vmag = dss.Bus.puVmagAngle()
                if vmag:
                    v_pu = vmag[0::2][:dss.Bus.NumNodes()]
                    voltages.extend(v_pu)

            valid_voltages = [v for v in voltages if 0.1 < v < 2.0]
            step_data["min_voltage_pu"] = min(valid_voltages) if valid_voltages else 0.0
            step_data["max_voltage_pu"] = max(valid_voltages) if valid_voltages else 0.0
            step_data["voltage_violations"] = sum(
                1 for v in valid_voltages if v < 0.95 or v > 1.05
            )
        else:
            step_data["source_power_kw"] = np.nan
            for feeder in FEEDER_HEAD_LINES:
                step_data[f"power_{feeder}_kw"] = np.nan
            step_data["min_voltage_pu"] = np.nan
            step_data["max_voltage_pu"] = np.nan
            step_data["voltage_violations"] = 0

        results.append(step_data)

    return results


def validate(target_date: str):
    """Run validation for a target date.

    Returns
    -------
    dict or None
        Structured results dict with keys: converged_steps, system_mae_mw,
        system_correlation, per-feeder MAE/correlation, voltage stats.
        Returns None if validation cannot run (missing data/OpenDSS).
    """
    logger.info(f"Validating simulation for date: {target_date}")

    results = {"date": target_date}

    # Load measured data
    if not LOAD_PROFILES_CLEANED.exists():
        logger.error(f"Cleaned load profiles not found: {LOAD_PROFILES_CLEANED}")
        logger.error("Run the pipeline first (python scripts/pipeline.py)")
        return None

    load_df = pd.read_csv(LOAD_PROFILES_CLEANED, parse_dates=["Timestamp"])
    target = pd.Timestamp(target_date)
    target_end = target + pd.Timedelta(days=1)

    # Get measured data for target date per feeder
    measured = {}
    for feeder in LOAD_FEEDERS:
        feeder_data = load_df[
            (load_df["Feeder"] == feeder) &
            (load_df["Timestamp"] >= target) &
            (load_df["Timestamp"] < target_end)
        ].sort_values("Timestamp")

        if len(feeder_data) > 0:
            feeder_data = feeder_data.drop_duplicates(subset="Timestamp", keep="last")
            expected_ts = pd.date_range(start=target, periods=96, freq="15min")
            aligned = feeder_data.set_index("Timestamp").sort_index()["MW"].reindex(
                expected_ts, method="nearest", tolerance="30min"
            ).interpolate().bfill().ffill()
            measured[feeder] = aligned.values[:96]
            logger.info(f"  {feeder}: {len(feeder_data)} raw points, "
                        f"MW range [{feeder_data['MW'].min():.2f}, {feeder_data['MW'].max():.2f}]")
        else:
            logger.warning(f"  {feeder}: No measured data for {target_date}")

    if not measured:
        logger.error("No measured data found for any feeder. Cannot validate.")
        return None

    # Run simulation
    sim_results = _try_opendss_simulation()

    if sim_results is None:
        logger.info("\n--- Validation Summary (data-only, no simulation) ---")
        logger.info(f"Measured data available for feeders: {list(measured.keys())}")
        for feeder, mw_values in measured.items():
            logger.info(f"  {feeder}: Peak={max(mw_values):.2f} MW, "
                        f"Min={min(mw_values):.2f} MW, "
                        f"Mean={np.mean(mw_values):.2f} MW")
        return None

    # Compile simulation results
    sim_df = pd.DataFrame(sim_results)

    logger.info("\n" + "=" * 70)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 70)

    converged_count = int(sim_df["converged"].sum())
    results["converged_steps"] = converged_count
    logger.info(f"Convergence: {converged_count}/96 steps converged")

    conv_df = sim_df[sim_df["converged"]].copy()

    if len(conv_df) == 0:
        logger.error("No steps converged! Cannot compute statistics.")
        return results

    logger.info(f"Source power range: "
                f"[{conv_df['source_power_kw'].min():.0f}, "
                f"{conv_df['source_power_kw'].max():.0f}] kW")

    # Voltage analysis
    logger.info(f"\nVoltage Analysis (converged steps):")
    min_v = float(conv_df['min_voltage_pu'].min())
    max_v = float(conv_df['max_voltage_pu'].max())
    total_violations = int(conv_df["voltage_violations"].sum())
    results["min_voltage_pu"] = min_v
    results["max_voltage_pu"] = max_v
    results["voltage_violations"] = total_violations
    logger.info(f"  Min voltage: {min_v:.4f} pu")
    logger.info(f"  Max voltage: {max_v:.4f} pu")
    logger.info(f"  Total voltage violations (0.95-1.05): {total_violations}")

    # ---- Per-feeder validation ----
    logger.info(f"\n--- Per-Feeder Validation ({converged_count}/96 converged) ---")
    mask = sim_df["converged"].values[:96]

    feeder_maes = []
    feeder_corrs = []
    for feeder in measured:
        col = f"power_{feeder}_kw"
        if col not in sim_df.columns:
            logger.warning(f"  {feeder}: No simulation data")
            continue

        sim_feeder_kw = sim_df[col].values[:96]
        meas_feeder_kw = measured[feeder] * 1000.0  # MW to kW

        sim_conv = sim_feeder_kw[mask]
        meas_conv = meas_feeder_kw[mask]

        # Drop NaN pairs
        valid = ~(np.isnan(sim_conv) | np.isnan(meas_conv))
        if valid.sum() < 3:
            logger.info(f"  {feeder}: Insufficient valid data ({valid.sum()} points)")
            continue

        sim_v = sim_conv[valid]
        meas_v = meas_conv[valid]

        mae_kw = np.mean(np.abs(sim_v - meas_v))
        mae_mw = float(mae_kw / 1000.0)
        mape = np.mean(np.abs(
            (sim_v - meas_v) / np.maximum(np.abs(meas_v), 100.0)
        )) * 100
        corr = float(np.corrcoef(sim_v, meas_v)[0, 1]) if len(sim_v) > 2 else np.nan

        feeder_maes.append(mae_mw)
        results[f"{feeder}_mae_mw"] = mae_mw
        results[f"{feeder}_correlation"] = corr
        if not np.isnan(corr):
            feeder_corrs.append(corr)

        status = "PASS" if mae_mw < 1.0 else "FAIL"
        corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
        logger.info(f"  {feeder}: MAE={mae_mw:.2f} MW [{status}], "
                    f"MAPE={mape:.1f}%, Corr={corr_str}")

    # ---- System-level validation (sum of feeder powers) ----
    logger.info(f"\n--- System-Level Validation (sum of feeder powers) ---")

    feeder_cols = [f"power_{f}_kw" for f in measured.keys()
                   if f"power_{f}_kw" in sim_df.columns]
    if feeder_cols:
        total_sim_kw = sim_df[feeder_cols].sum(axis=1).values[:96]
        total_meas_kw = sum(measured[f] for f in measured) * 1000.0

        sim_conv = total_sim_kw[mask]
        meas_conv = total_meas_kw[mask]

        valid = ~(np.isnan(sim_conv) | np.isnan(meas_conv))
        if valid.sum() > 2:
            sim_v = sim_conv[valid]
            meas_v = meas_conv[valid]

            mae = np.mean(np.abs(sim_v - meas_v))
            mape = np.mean(np.abs(
                (sim_v - meas_v) / np.maximum(np.abs(meas_v), 1.0)
            )) * 100
            corr = float(np.corrcoef(sim_v, meas_v)[0, 1])

            results["system_mae_mw"] = float(mae / 1000.0)
            results["system_correlation"] = corr

            logger.info(f"  Total feeder power range (sim): "
                        f"[{np.nanmin(sim_v):.0f}, {np.nanmax(sim_v):.0f}] kW")
            logger.info(f"  Total feeder power range (meas): "
                        f"[{np.nanmin(meas_v):.0f}, {np.nanmax(meas_v):.0f}] kW")
            logger.info(f"  MAE:         {mae:.0f} kW ({mae / 1000:.2f} MW) [target < 5000 kW]")
            logger.info(f"  MAPE:        {mape:.1f}% [target < 15%]")
            logger.info(f"  Correlation: {corr:.4f} [target > 0.85]")

            # Pass/Fail summary
            passes = []
            avg_feeder_mae = np.mean(feeder_maes) if feeder_maes else float('inf')
            avg_feeder_corr = np.mean(feeder_corrs) if feeder_corrs else 0.0
            passes.append(f"Avg Feeder MAE: {'PASS' if avg_feeder_mae < 1.0 else 'FAIL'} "
                          f"({avg_feeder_mae:.2f} MW)")
            passes.append(f"System MAE: {'PASS' if mae / 1000 < 5.0 else 'FAIL'} "
                          f"({mae / 1000:.2f} MW)")
            passes.append(f"System Corr: {'PASS' if corr > 0.85 else 'FAIL'} ({corr:.4f})")
            passes.append(f"Convergence: {'PASS' if converged_count >= 80 else 'FAIL'} "
                          f"({converged_count}/96)")

            logger.info(f"\n  Results: {', '.join(passes)}")

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2025-08-01")
    args = parser.parse_args()
    validate(args.date)
