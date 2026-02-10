"""
Generate OpenDSS LoadShape DSS file for solar irradiance.

Produces Solar_YYYYMMDD.dss in LoadShapes_RealData/ with a single shared
solar irradiance shape used by all PV systems.
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


def generate_solar_shapes(target_date: str, multipliers: dict = None) -> Path:
    """Generate solar irradiance LoadShape DSS file.

    Args:
        target_date: Date string "YYYY-MM-DD"
        multipliers: Optional pre-computed dict (from disaggregate()).

    Returns:
        Path to generated DSS file.
    """
    LOADSHAPES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.replace("-", "")

    # Get solar multipliers
    if multipliers and "Solar_Shape" in multipliers:
        mults = multipliers["Solar_Shape"]
    else:
        csv_path = DISAGGREGATED_DIR / f"Solar_Shape_{date_str}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mults = df["multiplier"].tolist()
        else:
            raise FileNotFoundError(f"No solar shape data found: {csv_path}")

    mult_str = _format_mult_array(mults)

    shape_name = f"Solar_{date_str}"
    dss_lines = [
        f"// ============================================================================",
        f"// SOLAR IRRADIANCE SHAPE - Real NASA Data for {target_date}",
        f"// Location: Jaffna, Sri Lanka (9.7423N, 80.0352E)",
        f"// Source: NASA/POWER CERES SYN1deg",
        f"// ============================================================================",
        f"",
        f"New Loadshape.{shape_name}",
        f"~   npts=96  interval=0.25",
        f"~   mult=[{mult_str}]",
        f"",
    ]

    out_path = LOADSHAPES_DIR / f"Solar_{date_str}.dss"
    out_path.write_text("\n".join(dss_lines), encoding="utf-8")
    logger.info(f"  Generated: {out_path.name}")

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_solar_shapes("2025-08-01")
