"""
Generate OpenDSS LoadShape DSS files for individual loads from disaggregated data.

Produces one DSS file per feeder (e.g. F06_20250801.dss) in LoadShapes_RealData/.
Each file contains LoadShape definitions for all loads on that feeder.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DISAGGREGATED_DIR, LOADSHAPES_DIR, FEEDER_LOADS, LOAD_FEEDERS

logger = logging.getLogger(__name__)


def _format_mult_array(values: list) -> str:
    """Format a list of multiplier values as a single-line DSS mult array string.

    OpenDSS requires the entire mult=[...] on one line (or using ~ continuation).
    Safest approach is a single line.
    """
    return ", ".join(f"{v:.4f}" for v in values)


def generate_load_shapes(target_date: str, multipliers: dict = None) -> list:
    """Generate per-feeder DSS LoadShape files.

    Args:
        target_date: Date string "YYYY-MM-DD"
        multipliers: Optional pre-computed multiplier dict (from disaggregate()).
                     If None, reads from DISAGGREGATED_DIR CSV files.

    Returns:
        List of generated DSS file paths.
    """
    LOADSHAPES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.replace("-", "")  # "20250801"

    generated_files = []

    for feeder in LOAD_FEEDERS:
        loads = FEEDER_LOADS[feeder]
        dss_lines = []
        dss_lines.append(f"// ============================================================================")
        dss_lines.append(f"// {feeder} LOAD SHAPES - Real Data for {target_date}")
        dss_lines.append(f"// Generated from CEB metering data + disaggregation")
        dss_lines.append(f"// ============================================================================")
        dss_lines.append("")

        for load_info in loads:
            name = load_info["name"]
            shape_name = f"{name}_{date_str}"

            # Get multiplier data
            if multipliers and name in multipliers:
                mults = multipliers[name]
            else:
                csv_path = DISAGGREGATED_DIR / f"{name}_{date_str}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    mults = df["multiplier"].tolist()
                else:
                    logger.warning(f"  No multiplier data for {name}, skipping")
                    continue

            if len(mults) != 96:
                logger.warning(f"  {name} has {len(mults)} values (expected 96), padding/trimming")
                if len(mults) < 96:
                    mults = mults + [mults[-1]] * (96 - len(mults))
                else:
                    mults = mults[:96]

            mult_str = _format_mult_array(mults)

            dss_lines.append(f"New Loadshape.{shape_name}")
            dss_lines.append(f"~   npts=96  interval=0.25")
            dss_lines.append(f"~   mult=[{mult_str}]")
            dss_lines.append("")

        # Write DSS file
        out_path = LOADSHAPES_DIR / f"{feeder}_{date_str}.dss"
        out_path.write_text("\n".join(dss_lines), encoding="utf-8")
        generated_files.append(out_path)
        logger.info(f"  Generated: {out_path.name} ({len(loads)} shapes)")

    return generated_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_load_shapes("2025-08-01")
