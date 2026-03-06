"""
Solar Forecast Service — loads trained ML models and produces 24-hour ahead
solar generation forecasts using a Random Forest autoregressive pipeline.

Models:
  - Random Forest (17 features) for 24h ahead forecasting
  - LSTM (24-step sequence of solar_generation_mw) for live single-step
  - GradientBoosting meta model (stacks RF + LSTM) for live single-step
  - MinMaxScaler for LSTM input/output normalization
"""
import logging
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature order expected by the RF model (must match training notebook)
RF_FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "T2M",
    "RH2M",
    "WS10M",
    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_3h",
    "rolling_6h",
    "rolling_24h_mean",
    "rolling_7day_mean",
]


class SolarForecastService:

    def __init__(self):
        self.rf_model = None
        self.lstm_model = None
        self.meta_model = None
        self.scaler = None  # MinMaxScaler for LSTM only
        self.weather_df: Optional[pd.DataFrame] = None
        self.solar_history: Optional[pd.Series] = None  # hourly solar_gen_mw
        self.is_loaded = False

        # Live simulation state
        self._lstm_window: deque = deque(maxlen=24)
        self._live_lag_1 = 0.0
        self._live_lag_24 = 0.0
        self._live_lag_168 = 0.0
        self._live_roll_3 = 0.0
        self._live_roll_6 = 0.0
        self._live_roll_24 = 0.0
        self._live_roll_168 = 0.0

    # ------------------------------------------------------------------ #
    #  Startup                                                            #
    # ------------------------------------------------------------------ #

    def load_models(self):
        """Load all model files and the NASA POWER weather CSV."""
        from config import settings

        models_dir: Path = settings.SOLAR_MODELS_DIR
        csv_path: Path = settings.NASA_POWER_CSV

        # --- Load model files ---
        rf_path = models_dir / "solar_random_fr_model.pkl"
        lstm_path = models_dir / "solar_lstm_model.keras"
        meta_path = models_dir / "solar_meta_model.pkl"
        scaler_path = models_dir / "solar_scaler.pkl"

        for p in [rf_path, lstm_path, meta_path, scaler_path]:
            if not p.exists():
                raise RuntimeError(f"Solar model file not found: {p}")

        # RF model is required for 24h forecast
        for p in [rf_path, scaler_path]:
            if not p.exists():
                raise RuntimeError(f"Solar model file not found: {p}")

        self.rf_model = joblib.load(rf_path)
        logger.info(f"  RF model loaded: {rf_path}")

        self.scaler = joblib.load(scaler_path)
        logger.info(f"  Scaler loaded: {scaler_path}")

        # Meta model — optional (sklearn version mismatch may cause failure)
        try:
            self.meta_model = joblib.load(meta_path)
            logger.info(f"  Meta model loaded: {meta_path}")
        except Exception as e:
            logger.warning(f"  Meta model failed to load (RF-only fallback): {e}")
            self.meta_model = None

        # LSTM model — optional
        try:
            import tensorflow as tf
            self.lstm_model = tf.keras.models.load_model(str(lstm_path))
            logger.info(f"  LSTM model loaded: {lstm_path}")
        except Exception as e:
            logger.warning(f"  LSTM model failed to load (RF-only fallback): {e}")
            self.lstm_model = None

        # --- Load weather data ---
        if not csv_path.exists():
            raise RuntimeError(f"NASA POWER CSV not found: {csv_path}")

        # Count header lines (between -BEGIN HEADER- and -END HEADER-)
        skip_rows = 0
        with open(csv_path, "r") as f:
            for line in f:
                skip_rows += 1
                if "-END HEADER-" in line:
                    break

        raw = pd.read_csv(csv_path, skiprows=skip_rows)
        raw["datetime"] = pd.to_datetime(
            {"year": raw["YEAR"], "month": raw["MO"], "day": raw["DY"], "hour": raw["HR"]}
        )
        raw.set_index("datetime", inplace=True)
        raw = raw[["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
                    "T2M", "RH2M", "WS10M"]].copy()

        # Replace -999 (missing) with 0
        raw.replace(-999, 0, inplace=True)

        self.weather_df = raw

        # Pre-compute historical solar_generation_mw for lag initialization
        cap = settings.INSTALLED_CAPACITY_MW
        pr = settings.PERFORMANCE_RATIO
        self.solar_history = (raw["ALLSKY_SFC_SW_DWN"] / 1000.0) * cap * pr
        self.solar_history.name = "solar_generation_mw"

        logger.info(f"  Weather data loaded: {len(raw)} hourly rows "
                    f"({raw.index.min()} to {raw.index.max()})")

        self.is_loaded = True

    # ------------------------------------------------------------------ #
    #  24-Hour Ahead Forecast (RF autoregressive)                         #
    # ------------------------------------------------------------------ #

    def forecast_24h(self, target_date: str, start_hour: int = 0) -> List[Dict]:
        """
        Produce a 24-hour ahead solar generation forecast for *target_date*.

        Returns a list of 24 dicts:
          {timestamp: str, predicted_mw: float, hour: int}
        """
        if not self.is_loaded:
            raise RuntimeError("Solar forecast service not loaded")

        date = pd.Timestamp(target_date)
        start = pd.Timestamp(f"{target_date} {start_hour:02d}:00:00")
        end = start + pd.Timedelta(hours=23)

        # Get weather rows for the 24 forecast hours
        weather_slice = self.weather_df.loc[start:end]
        if len(weather_slice) < 24:
            available_start = self.weather_df.index.min().date()
            available_end = self.weather_df.index.max().date()
            raise ValueError(
                f"Weather data not available for {target_date}. "
                f"Available range: {available_start} to {available_end}"
            )
        weather_slice = weather_slice.iloc[:24]

        # Initialize lags from historical solar_generation_mw
        hist = self.solar_history
        lag_1 = self._safe_lag(hist, start, 1)
        lag_24 = self._safe_lag(hist, start, 24)
        lag_168 = self._safe_lag(hist, start, 168)

        roll_3 = self._safe_rolling_mean(hist, start, 3)
        roll_6 = self._safe_rolling_mean(hist, start, 6)
        roll_24 = self._safe_rolling_mean(hist, start, 24)
        roll_168 = self._safe_rolling_mean(hist, start, 168)

        # RF autoregressive loop
        predictions = []
        for i in range(24):
            row = weather_slice.iloc[i]
            ts = weather_slice.index[i]
            hr = ts.hour
            mo = ts.month

            features = {
                "ALLSKY_SFC_SW_DWN": row["ALLSKY_SFC_SW_DWN"],
                "ALLSKY_SFC_SW_DNI": row["ALLSKY_SFC_SW_DNI"],
                "ALLSKY_SFC_SW_DIFF": row["ALLSKY_SFC_SW_DIFF"],
                "T2M": row["T2M"],
                "RH2M": row["RH2M"],
                "WS10M": row["WS10M"],
                "hour": hr,
                "month": mo,
                "hour_sin": math.sin(2 * math.pi * hr / 24),
                "hour_cos": math.cos(2 * math.pi * hr / 24),
                "lag_1": lag_1,
                "lag_24": lag_24,
                "lag_168": lag_168,
                "rolling_3h": roll_3,
                "rolling_6h": roll_6,
                "rolling_24h_mean": roll_24,
                "rolling_7day_mean": roll_168,
            }

            X = pd.DataFrame([features], columns=RF_FEATURES)
            rf_pred = float(self.rf_model.predict(X)[0])

            # Update lags / rolling (matching notebook logic)
            lag_168 = lag_24
            lag_24 = lag_1
            lag_1 = rf_pred

            roll_3 = (roll_3 * 2 + rf_pred) / 3
            roll_6 = (roll_6 * 5 + rf_pred) / 6
            roll_24 = (roll_24 * 23 + rf_pred) / 24
            roll_168 = (roll_168 * 167 + rf_pred) / 168

            predictions.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "hour": hr,
                "predicted_mw": rf_pred,
            })

        # Post-processing: night zero, sunrise/sunset smoothing
        predictions = self._post_process(predictions)

        return predictions

    # ------------------------------------------------------------------ #
    #  Live Single-Step Prediction (RF + optional LSTM + Meta)            #
    # ------------------------------------------------------------------ #

    def predict_single_step(self, weather_row: Dict, actual_solar_mw: float) -> Dict:
        """
        One-step-ahead prediction for live simulation mode.
        Updates internal state (lags, rolling, LSTM window).
        """
        if not self.is_loaded:
            raise RuntimeError("Solar forecast service not loaded")

        hr = weather_row.get("hour", 0)
        mo = weather_row.get("month", 1)

        features = {
            "ALLSKY_SFC_SW_DWN": weather_row.get("ALLSKY_SFC_SW_DWN", 0),
            "ALLSKY_SFC_SW_DNI": weather_row.get("ALLSKY_SFC_SW_DNI", 0),
            "ALLSKY_SFC_SW_DIFF": weather_row.get("ALLSKY_SFC_SW_DIFF", 0),
            "T2M": weather_row.get("T2M", 0),
            "RH2M": weather_row.get("RH2M", 0),
            "WS10M": weather_row.get("WS10M", 0),
            "hour": hr,
            "month": mo,
            "hour_sin": math.sin(2 * math.pi * hr / 24),
            "hour_cos": math.cos(2 * math.pi * hr / 24),
            "lag_1": self._live_lag_1,
            "lag_24": self._live_lag_24,
            "lag_168": self._live_lag_168,
            "rolling_3h": self._live_roll_3,
            "rolling_6h": self._live_roll_6,
            "rolling_24h_mean": self._live_roll_24,
            "rolling_7day_mean": self._live_roll_168,
        }

        X = pd.DataFrame([features], columns=RF_FEATURES)
        rf_pred = float(self.rf_model.predict(X)[0])

        # LSTM prediction (if model loaded and window full)
        lstm_pred = None
        meta_pred = None
        self._lstm_window.append(actual_solar_mw)

        if self.lstm_model is not None and len(self._lstm_window) == 24:
            try:
                window_arr = np.array(list(self._lstm_window)).reshape(-1, 1)
                scaled = self.scaler.transform(window_arr)
                X_lstm = scaled.reshape(1, 24, 1)
                lstm_scaled = self.lstm_model.predict(X_lstm, verbose=0)
                lstm_pred = float(self.scaler.inverse_transform(lstm_scaled)[0, 0])

                # Meta model stacking (if available)
                if self.meta_model is not None:
                    meta_input = np.array([[rf_pred, lstm_pred]])
                    meta_pred = float(self.meta_model.predict(meta_input)[0])
            except Exception as e:
                logger.warning(f"LSTM/Meta prediction failed: {e}")

        # Update live state using actual values
        self._live_lag_168 = self._live_lag_24
        self._live_lag_24 = self._live_lag_1
        self._live_lag_1 = actual_solar_mw

        self._live_roll_3 = (self._live_roll_3 * 2 + actual_solar_mw) / 3
        self._live_roll_6 = (self._live_roll_6 * 5 + actual_solar_mw) / 6
        self._live_roll_24 = (self._live_roll_24 * 23 + actual_solar_mw) / 24
        self._live_roll_168 = (self._live_roll_168 * 167 + actual_solar_mw) / 168

        final_pred = meta_pred if meta_pred is not None else rf_pred

        return {
            "predicted_mw": round(max(0, final_pred), 2),
            "rf_pred": round(rf_pred, 2),
            "lstm_pred": round(lstm_pred, 2) if lstm_pred is not None else None,
            "meta_pred": round(meta_pred, 2) if meta_pred is not None else None,
            "actual_mw": round(actual_solar_mw, 2),
        }

    def reset_live_state(self):
        """Reset live simulation state for a new simulation run."""
        self._lstm_window.clear()
        self._live_lag_1 = 0.0
        self._live_lag_24 = 0.0
        self._live_lag_168 = 0.0
        self._live_roll_3 = 0.0
        self._live_roll_6 = 0.0
        self._live_roll_24 = 0.0
        self._live_roll_168 = 0.0

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _safe_lag(self, series: pd.Series, ref_time: pd.Timestamp, hours_back: int) -> float:
        target = ref_time - pd.Timedelta(hours=hours_back)
        if target in series.index:
            val = series.loc[target]
            return float(val) if not pd.isna(val) else 0.0
        return 0.0

    def _safe_rolling_mean(self, series: pd.Series, ref_time: pd.Timestamp, window: int) -> float:
        end = ref_time - pd.Timedelta(hours=1)
        start = ref_time - pd.Timedelta(hours=window)
        chunk = series.loc[start:end]
        if len(chunk) > 0:
            return float(chunk.mean())
        return 0.0

    def _post_process(self, predictions: List[Dict]) -> List[Dict]:
        """Apply night zeroing and sunrise/sunset smoothing (matching notebook)."""
        for p in predictions:
            hr = p["hour"]
            mw = p["predicted_mw"]

            # Night hours: force to zero
            if hr < 6 or hr > 16:
                mw = 0.0
            # Sunrise smoothing
            elif hr == 6:
                mw *= 0.3
            elif hr == 7:
                mw *= 0.6
            # Sunset smoothing
            elif hr == 15:
                mw *= 0.6
            elif hr == 16:
                mw *= 0.3

            p["predicted_mw"] = round(max(0, mw), 2)

        return predictions

    def get_actual_solar_for_date(self, date_str: str) -> List[float]:
        """Return 24 hourly actual solar_generation_mw values for a date from NASA data."""
        if not self.is_loaded:
            raise RuntimeError("Solar forecast service not loaded")

        start = pd.Timestamp(f"{date_str} 00:00:00")
        end = pd.Timestamp(f"{date_str} 23:00:00")

        chunk = self.solar_history.loc[start:end]
        if len(chunk) < 24:
            raise ValueError(f"Actual solar data not available for {date_str}")

        return [round(float(v), 2) for v in chunk.iloc[:24].values]

    def get_day_data(self, date_str: str) -> Dict:
        """
        Return actual solar for *date_str* and predicted solar for the next day.
        Used by the simulation-driven forecasting page.
        """
        if not self.is_loaded:
            raise RuntimeError("Solar forecast service not loaded")

        actual_mw = self.get_actual_solar_for_date(date_str)

        # Compute next date
        next_date = (pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Try to forecast next day; if out of range, return empty
        predicted_next_day_mw = []
        try:
            preds = self.forecast_24h(next_date)
            predicted_next_day_mw = [p["predicted_mw"] for p in preds]
        except ValueError:
            logger.info(f"Cannot forecast {next_date} — beyond weather data range")

        return {
            "date": date_str,
            "next_date": next_date,
            "actual_mw": actual_mw,
            "predicted_next_day_mw": predicted_next_day_mw,
        }

    def get_available_date_range(self) -> Dict:
        """Return the date range covered by the weather data."""
        if self.weather_df is None:
            return {"start": None, "end": None}
        return {
            "start": str(self.weather_df.index.min().date()),
            "end": str(self.weather_df.index.max().date()),
        }


# Singleton instance
solar_forecast_service = SolarForecastService()
