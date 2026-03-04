"""
Report Generator — Produce summary comparison table as CSV and formatted text.
Optionally generates a PDF report if reportlab is available.
IT22577924 — Karunanayake K.P.A.W.
"""
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

RESEARCH_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR  = RESEARCH_DIR / "results" / "reports"


def generate_text_report(results: list, output_path: Path = None) -> str:
    """
    Generate a formatted text comparison table.

    Parameters
    ----------
    results : list[dict]  — from benchmark.py, each with model name + metrics

    Returns
    -------
    str : formatted report text (also written to output_path if provided)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 90,
        "  FAULT DIAGNOSTICS MODEL COMPARISON REPORT",
        "  IT22577924 — Chunnakam GSS — Karunanayake K.P.A.W.",
        f"  Generated: {timestamp}",
        "=" * 90,
        "",
        "Task: Fault Type Classification (6 classes: Normal, LG, LL, LLG, LLL, HIF)",
        "",
        f"{'Model':<32} {'Accuracy':>10} {'F1-Macro':>10} {'MCC':>10} {'Prec':>10} {'Recall':>10}",
        "-" * 85,
    ]

    for r in results:
        lines.append(
            f"{r['model']:<32} "
            f"{r.get('type_accuracy', 0):>10.4f} "
            f"{r.get('type_f1_macro', 0):>10.4f} "
            f"{r.get('type_mcc', 0):>10.4f} "
            f"{r.get('type_precision', 0):>10.4f} "
            f"{r.get('type_recall', 0):>10.4f}"
        )

    lines += [
        "-" * 85, "",
        "Task: Fault Detection (Binary)",
        "",
        f"{'Model':<32} {'Accuracy':>10} {'F1-Macro':>10} {'MCC':>10}",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r['model']:<32} "
            f"{r.get('detection_accuracy', 0):>10.4f} "
            f"{r.get('detection_f1_macro', 0):>10.4f} "
            f"{r.get('detection_mcc', 0):>10.4f}"
        )

    lines += [
        "-" * 60, "",
        "Task: Fault Location (Bus Index)",
        "",
        f"{'Model':<32} {'Accuracy':>10} {'F1-Macro':>10}",
        "-" * 55,
    ]
    for r in results:
        lines.append(
            f"{r['model']:<32} "
            f"{r.get('location_accuracy', 0):>10.4f} "
            f"{r.get('location_f1_macro', 0):>10.4f}"
        )

    lines += ["", "=" * 90]
    text = "\n".join(lines)

    if output_path is not None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Text report saved → {output_path}")

    return text


def generate_csv_report(results: list, output_path: Path = None) -> Path:
    """Write all metrics to CSV for spreadsheet analysis."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = REPORTS_DIR / "full_metrics_report.csv"

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    all_keys.discard("model")
    fieldnames = ["model"] + sorted(all_keys)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for k in fieldnames:
                v = r.get(k, "")
                row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
            writer.writerow(row)

    logger.info(f"CSV report saved → {output_path}")
    return output_path


def load_benchmark_results() -> list:
    """Load results from benchmark CSV if it exists."""
    csv_path = REPORTS_DIR / "benchmark_comparison.csv"
    if not csv_path.exists():
        logger.warning(f"Benchmark CSV not found: {csv_path}")
        return []

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = v
            results.append(d)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = load_benchmark_results()
    if results:
        text = generate_text_report(results, REPORTS_DIR / "comparison_report.txt")
        print(text)
        generate_csv_report(results, REPORTS_DIR / "full_metrics_report.csv")
    else:
        print("No benchmark results found. Run benchmark.py first.")
