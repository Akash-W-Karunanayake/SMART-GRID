"""
DSS date reference updater.

Replaces all _YYYYMMDD date patterns in OpenDSS master files so the
simulation references match the target date. Idempotent — works regardless
of which date is currently embedded.
"""
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches an 8-digit date string preceded by underscore.
# Captures the date portion; the underscore is kept via lookbehind.
_DATE_PATTERN = re.compile(r'(?<=_)(\d{8})(?=[\s.\n]|$)')


def update_dss_references(dss_files: list[Path], new_date_str: str) -> int:
    """Rewrite all _YYYYMMDD references in *dss_files* to *new_date_str*.

    Parameters
    ----------
    dss_files : list[Path]
        DSS files to update (Households.dss, RooftopSolar.dss, etc.).
    new_date_str : str
        Target date in YYYY-MM-DD format (e.g. "2025-07-15").

    Returns
    -------
    int
        Total number of replacements made across all files.
    """
    compact = new_date_str.replace("-", "")  # "20250715"
    total_replacements = 0

    for dss_path in dss_files:
        if not dss_path.exists():
            logger.warning(f"DSS file not found, skipping: {dss_path}")
            continue

        text = dss_path.read_text(encoding="utf-8")
        new_text, count = _DATE_PATTERN.subn(compact, text)

        if count > 0:
            dss_path.write_text(new_text, encoding="utf-8")
            logger.info(f"  Updated {count} date references in {dss_path.name}")
        else:
            logger.debug(f"  No date references found in {dss_path.name}")

        total_replacements += count

    logger.info(f"  Total: {total_replacements} date references updated to {compact}")
    return total_replacements
