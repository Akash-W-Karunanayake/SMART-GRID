"""
Validation Test A — Unit test the solar forecast pipeline in isolation.
Loads models, runs a 24-hour forecast, checks output sanity.
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.solar_forecast_service import solar_forecast_service


def test_forecast_24h():
    print("=" * 60)
    print("Solar Forecast Pipeline — Validation Test A")
    print("=" * 60)

    # Step 1: Load models
    print("\n[1] Loading models...")
    try:
        solar_forecast_service.load_models()
        print("    PASS — All models loaded successfully")
    except Exception as e:
        print(f"    FAIL — Model loading error: {e}")
        return False

    # Step 2: Check available date range
    date_range = solar_forecast_service.get_available_date_range()
    print(f"    Weather data range: {date_range['start']} to {date_range['end']}")

    # Step 3: Run 24-hour forecast for a mid-range date
    test_date = "2025-07-15"
    print(f"\n[2] Running 24h forecast for {test_date}...")
    try:
        predictions = solar_forecast_service.forecast_24h(test_date)
    except Exception as e:
        print(f"    FAIL — Forecast error: {e}")
        return False

    # Step 4: Validate output
    print(f"    Points returned: {len(predictions)}")
    if len(predictions) != 24:
        print(f"    FAIL — Expected 24 points, got {len(predictions)}")
        return False

    all_finite = all(
        isinstance(p["predicted_mw"], (int, float)) and p["predicted_mw"] >= 0
        for p in predictions
    )
    if not all_finite:
        print("    FAIL — Some predictions are not finite or negative")
        return False

    # Check night hours are zero
    night_ok = all(
        p["predicted_mw"] == 0
        for p in predictions
        if p["hour"] < 6 or p["hour"] > 16
    )
    if not night_ok:
        print("    WARN — Some night hours are non-zero (expected 0)")

    # Check peak is plausible (0-45 MW for 53.4 MW installed capacity)
    peak = max(p["predicted_mw"] for p in predictions)
    if peak > 50:
        print(f"    WARN — Peak prediction {peak:.2f} MW seems high")
    elif peak < 1:
        print(f"    WARN — Peak prediction {peak:.2f} MW seems too low")

    # Print all 24 predictions
    print(f"\n[3] 24-Hour Forecast Results:")
    print(f"    {'Hour':>6} {'Timestamp':>18} {'Predicted MW':>14}")
    print(f"    {'-'*6} {'-'*18} {'-'*14}")
    for p in predictions:
        print(f"    {p['hour']:>6} {p['timestamp']:>18} {p['predicted_mw']:>14.2f}")

    print(f"\n    Peak: {peak:.2f} MW")
    total_energy = sum(p["predicted_mw"] for p in predictions)
    print(f"    Total daily energy: {total_energy:.2f} MWh")

    print("\n" + "=" * 60)
    print("PASS — All validation checks passed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_forecast_24h()
    sys.exit(0 if success else 1)
