"""
Generate OpenDSS LoadShape DSS file for UJPS thermal generators.

Produces UJPS_YYYYMMDD.dss in LoadShapes_RealData/.
All 3 UJPS generators share the same dispatch shape.
"""
import pandas as pd
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DISAGGREGATED_DIR, LOADSHAPES_DIR

logger = logging.getLogger(__name__)


def _format_mult_array(values: list) -> str:
    """Format a list of multiplier values as a single-line DSS mult array string."""
    return ", ".join(f"{v:.4f}" for v in values)


def generate_ujps_shapes(target_date: str, multipliers: dict = None) -> Path:
    """Generate UJPS thermal dispatch LoadShape DSS file.

    Args:
        target_date: Date string "YYYY-MM-DD"
        multipliers: Optional pre-computed dict (from disaggregate()).

    Returns:
        Path to generated DSS file.
    """
    LOADSHAPES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.replace("-", "")

    # Get UJPS multipliers (all 3 generators share same shape)
    if multipliers and "UJPS_Gen1" in multipliers:
        mults = multipliers["UJPS_Gen1"]
    else:
        csv_path = DISAGGREGATED_DIR / f"UJPS_Gen1_{date_str}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mults = df["multiplier"].tolist()
        else:
            raise FileNotFoundError(f"No UJPS shape data found: {csv_path}")

    mult_str = _format_mult_array(mults)

    shape_name = f"UJPS_{date_str}"
    dss_lines = [
        f"// ============================================================================",
        f"// UJPS THERMAL GENERATOR SHAPE - Real Data for {target_date}",
        f"// Derived from CEB F05 metering data (net export = UJPS generation)",
        f"// Shared by all 3 UJPS generators (8 MW each)",
        f"// ============================================================================",
        f"",
        f"New Loadshape.{shape_name}",
        f"~   npts=96  interval=0.25",
        f"~   mult=[{mult_str}]",
        f"",
    ]

    out_path = LOADSHAPES_DIR / f"UJPS_{date_str}.dss"
    out_path.write_text("\n".join(dss_lines), encoding="utf-8")
    logger.info(f"  Generated: {out_path.name}")

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_ujps_shapes("2025-08-01")
