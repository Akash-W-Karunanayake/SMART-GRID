"""
Pipeline orchestrator for real data integration.

Usage:
    python scripts/pipeline.py --date 2025-08-01
    python scripts/pipeline.py --date 2025-08-01 --validate
    python scripts/pipeline.py --date 2025-07-15 --skip-preprocessing
    python scripts/pipeline.py --start 2025-05-01 --end 2025-08-31 --validate
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure scripts/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_DATE, LOAD_PROFILES_CLEANED, SOLAR_15MIN, WIND_15MIN,
    DSS_DATE_FILES, RESULTS_DIR,
)
from dss_date_updater import update_dss_references

logger = logging.getLogger(__name__)


def run_preprocessing():
    """Phase 1: Parse raw data files."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 70)

    from data_preprocessing.parse_load_profiles import parse_all_load_profiles
    from data_preprocessing.parse_solar import parse_solar
    from data_preprocessing.parse_wind import parse_wind

    logger.info("\n--- 1a. Parsing load profiles ---")
    parse_all_load_profiles()

    logger.info("\n--- 1b. Parsing solar data ---")
    parse_solar()

    logger.info("\n--- 1c. Parsing wind data ---")
    parse_wind()


def run_disaggregation(target_date: str) -> dict:
    """Phase 2: Disaggregate feeder loads."""
    logger.info("=" * 70)
    logger.info("PHASE 2: LOAD DISAGGREGATION")
    logger.info("=" * 70)

    from loadshape_generation.disaggregate import disaggregate
    return disaggregate(target_date)


def run_shape_generation(target_date: str, multipliers: dict):
    """Phase 3: Generate OpenDSS LoadShape files."""
    logger.info("=" * 70)
    logger.info("PHASE 3: LOADSHAPE GENERATION")
    logger.info("=" * 70)

    from loadshape_generation.generate_load_shapes import generate_load_shapes
    from loadshape_generation.generate_solar_shapes import generate_solar_shapes
    from loadshape_generation.generate_wind_shapes import generate_wind_shapes
    from loadshape_generation.generate_ujps_shapes import generate_ujps_shapes

    logger.info("\n--- 3a. Generating load shapes ---")
    generate_load_shapes(target_date, multipliers)

    logger.info("\n--- 3b. Generating solar shape ---")
    generate_solar_shapes(target_date, multipliers)

    logger.info("\n--- 3c. Generating wind shape ---")
    generate_wind_shapes(target_date, multipliers)

    logger.info("\n--- 3d. Generating UJPS shape ---")
    generate_ujps_shapes(target_date, multipliers)


def run_dss_update(target_date: str):
    """Phase 4: Update date references in DSS master files."""
    logger.info("=" * 70)
    logger.info("PHASE 4: DSS DATE UPDATE")
    logger.info("=" * 70)

    count = update_dss_references(DSS_DATE_FILES, target_date)
    logger.info(f"  Updated {count} references to {target_date}")


def run_validation(target_date: str):
    """Phase 6: Validate simulation results against measured data.

    Returns
    -------
    dict or None
        Structured results from validate(), or None.
    """
    logger.info("=" * 70)
    logger.info("PHASE 6: VALIDATION")
    logger.info("=" * 70)

    from validation.validate_results import validate
    return validate(target_date)


def run_single_day(target_date: str, validate: bool) -> dict:
    """Run Phases 2-4 (and optionally 6) for a single date.

    Parameters
    ----------
    target_date : str
        Date in YYYY-MM-DD format.
    validate : bool
        Whether to run Phase 6 validation.

    Returns
    -------
    dict
        Results dict with at least 'date' and 'status' keys.
    """
    result = {"date": target_date, "status": "success"}

    try:
        # Phase 2: Disaggregation
        multipliers = run_disaggregation(target_date)

        # Phase 3: Shape generation
        run_shape_generation(target_date, multipliers)

        # Phase 4: DSS date update
        run_dss_update(target_date)

        # Phase 6: Validation (optional)
        if validate:
            val_results = run_validation(target_date)
            if val_results:
                result.update(val_results)

    except Exception as e:
        logger.error(f"Error processing {target_date}: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


def run_multi_day(start_date: str, end_date: str, validate: bool) -> list[dict]:
    """Run the pipeline for each day in a date range.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format (inclusive).
    end_date : str
        End date in YYYY-MM-DD format (inclusive).
    validate : bool
        Whether to run validation for each day.

    Returns
    -------
    list[dict]
        List of per-day results dicts.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if end < start:
        logger.error(f"End date {end_date} is before start date {start_date}")
        sys.exit(1)

    total_days = (end - start).days + 1
    logger.info(f"Multi-day mode: {start_date} to {end_date} ({total_days} days)")

    all_results = []
    current = start
    day_num = 0

    while current <= end:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"# DAY {day_num}/{total_days}: {date_str}")
        logger.info("#" * 70)

        result = run_single_day(date_str, validate)
        all_results.append(result)

        status = result["status"]
        if status == "success":
            logger.info(f"Day {date_str}: COMPLETED")
        else:
            logger.warning(f"Day {date_str}: {status.upper()} - {result.get('error', '')}")

        current += timedelta(days=1)

    return all_results


def _save_multi_day_results(results: list[dict], start_date: str, end_date: str):
    """Save aggregated multi-day results to CSV.

    Parameters
    ----------
    results : list[dict]
        Per-day results dicts from run_multi_day().
    start_date, end_date : str
        Date range strings for the filename.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start_compact = start_date.replace("-", "")
    end_compact = end_date.replace("-", "")
    csv_path = RESULTS_DIR / f"multiday_{start_compact}_to_{end_compact}.csv"

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Multi-day results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Real Data Integration Pipeline")

    # Date selection: either --date OR --start/--end
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date", default=None,
        help=f"Target simulation date (YYYY-MM-DD). Default: {DEFAULT_DATE}",
    )
    date_group.add_argument(
        "--start", default=None,
        help="Start date for multi-day mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date for multi-day mode (YYYY-MM-DD). Required with --start.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after pipeline",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip Phase 1 (use existing processed data)",
    )
    args = parser.parse_args()

    # Validate multi-day args
    if args.start and not args.end:
        parser.error("--end is required when using --start")
    if args.end and not args.start:
        parser.error("--start is required when using --end")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    multi_day = args.start is not None
    target_date = args.date if args.date else DEFAULT_DATE

    logger.info(f"Real Data Integration Pipeline")
    if multi_day:
        logger.info(f"Mode: Multi-day ({args.start} to {args.end})")
    else:
        logger.info(f"Target date: {target_date}")
    logger.info("")

    # Phase 1: Preprocessing (runs once)
    if not args.skip_preprocessing:
        run_preprocessing()
    else:
        # Verify processed files exist
        for f in [LOAD_PROFILES_CLEANED, SOLAR_15MIN, WIND_15MIN]:
            if not f.exists():
                logger.error(f"Missing processed file: {f}")
                logger.error("Run without --skip-preprocessing first")
                sys.exit(1)
        logger.info("Skipping Phase 1 (using existing processed data)")

    if multi_day:
        # Multi-day mode
        results = run_multi_day(args.start, args.end, args.validate)
        _save_multi_day_results(results, args.start, args.end)

        # Print summary
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "success")
        failed = total - passed
        logger.info("")
        logger.info("=" * 70)
        logger.info("MULTI-DAY PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Total days:  {total}")
        logger.info(f"  Succeeded:   {passed}")
        logger.info(f"  Failed:      {failed}")
        if failed > 0:
            for r in results:
                if r["status"] != "success":
                    logger.info(f"    {r['date']}: {r.get('error', 'unknown error')}")
    else:
        # Single-day mode
        # Phase 2: Disaggregation
        multipliers = run_disaggregation(target_date)

        # Phase 3: Shape generation
        run_shape_generation(target_date, multipliers)

        # Phase 4: DSS date update
        run_dss_update(target_date)

        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Generated LoadShape files in LoadShapes_RealData/")
        logger.info(f"DSS files have been modified to reference real data shapes")

        # Phase 6: Validation (optional)
        if args.validate:
            run_validation(target_date)


if __name__ == "__main__":
    main()
