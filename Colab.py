"""
=============================================================
 AQI Prediction System — Complete ML Pipeline
 Features:
   - Predicts AQI for next 7 days (per state/area)
   - Multi-model comparison (RandomForest, GradientBoosting,
     LinearRegression, Ridge)
   - OpenWeatherMap Air Pollution API integration
     (CO, NO, NO2, O3, SO2, NH3, PM2.5, PM10)
   - Full evaluation metrics & feature importance
   - Save / load model artifacts
=============================================================
"""

import os
import json
import warnings
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH          = "aqi.csv"           # path to your uploaded CSV
OPENWEATHER_API_KEY = "YOUR_API_KEY"   # ← replace with real key
MODEL_DIR         = "models"
PREDICTION_DAYS   = 7

# AQI category thresholds (India CPCB standard)
AQI_CATEGORIES = {
    (0,   50):  "Good",
    (51,  100): "Satisfactory",
    (101, 200): "Moderate",
    (201, 300): "Poor",
    (301, 400): "Very Poor",
    (401, 500): "Severe",
}

def aqi_category(val):
    for (lo, hi), label in AQI_CATEGORIES.items():
        if lo <= val <= hi:
            return label
    return "Severe"


# ═════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════
class AQIDataProcessor:
    """Load, clean, and feature-engineer the AQI CSV dataset."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.le_state   = LabelEncoder()
        self.le_area    = LabelEncoder()
        self.le_poll    = LabelEncoder()
        self.scaler     = StandardScaler()

    def load(self) -> pd.DataFrame:
        print("📂 Loading data …")
        df = pd.read_csv(self.csv_path)
        print(f"   Rows: {len(df):,}  |  Cols: {df.shape[1]}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🧹 Cleaning …")
        df = df.copy()

        # Parse date
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df.dropna(subset=["date", "aqi_value", "state", "area"], inplace=True)

        # Remove impossible AQI values
        df = df[(df["aqi_value"] >= 0) & (df["aqi_value"] <= 500)]

        # Clean text fields
        for col in ["state", "area", "prominent_pollutants"]:
            df[col] = df[col].astype(str).str.strip().str.title()

        print(f"   Clean rows: {len(df):,}")
        return df.sort_values("date").reset_index(drop=True)

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        print("⚙️  Engineering features …")
        df = df.copy()

        # ── Time features ──────────────────────────────────────────
        df["year"]          = df["date"].dt.year
        df["month"]         = df["date"].dt.month
        df["day_of_week"]   = df["date"].dt.dayofweek
        df["day_of_year"]   = df["date"].dt.dayofyear
        df["quarter"]       = df["date"].dt.quarter
        df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
        df["season"]        = df["month"].map(
            {12: 0, 1: 0, 2: 0,   # Winter
             3: 1, 4: 1, 5: 1,    # Spring
             6: 2, 7: 2, 8: 2,    # Monsoon
             9: 3, 10: 3, 11: 3}  # Autumn
        )

        # ── Lag / rolling features (per area) ─────────────────────
        df.sort_values(["area", "date"], inplace=True)
        grp = df.groupby("area")["aqi_value"]

        df["aqi_lag_1"]    = grp.shift(1)
        df["aqi_lag_7"]    = grp.shift(7)
        df["aqi_lag_30"]   = grp.shift(30)
        df["aqi_roll_7d"]  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
        df["aqi_roll_30d"] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        df["aqi_std_7d"]   = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).std().fillna(0))
        df["aqi_trend"]    = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) > 1 else 0
        ))

        # ── Monitoring stations ────────────────────────────────────
        df["monitoring_stations"] = df["number_of_monitoring_stations"].fillna(1)

        # ── Encode categoricals ────────────────────────────────────
        df["state_enc"] = self.le_state.fit_transform(df["state"])
        df["area_enc"]  = self.le_area.fit_transform(df["area"])
        df["poll_enc"]  = self.le_poll.fit_transform(
            df["prominent_pollutants"].fillna("Unknown")
        )

        df.dropna(subset=["aqi_lag_1"], inplace=True)
        print(f"   Feature-rich rows: {len(df):,}")
        return df.reset_index(drop=True)

    def get_feature_cols(self):
        return [
            "year", "month", "day_of_week", "day_of_year",
            "quarter", "is_weekend", "season",
            "state_enc", "area_enc", "poll_enc",
            "monitoring_stations",
            "aqi_lag_1", "aqi_lag_7", "aqi_lag_30",
            "aqi_roll_7d", "aqi_roll_30d", "aqi_std_7d", "aqi_trend"
        ]


# ═════════════════════════════════════════════
# 2. OPENWEATHERMAP AIR POLLUTION API
# ═════════════════════════════════════════════
class AirPollutionAPI:
    """
    Fetch real-time & forecast pollution data from OpenWeatherMap.
    Endpoint: https://api.openweathermap.org/data/2.5/air_pollution
    Docs:     https://openweathermap.org/api/air-pollution
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, endpoint: str, params: dict) -> dict:
        params["appid"] = self.api_key
        r = requests.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def current(self, lat: float, lon: float) -> dict:
        """Current air quality at (lat, lon)."""
        return self._parse(self._get("air_pollution", {"lat": lat, "lon": lon}))

    def forecast(self, lat: float, lon: float) -> list[dict]:
        """5-day / 1-hour forecast air quality."""
        data = self._get("air_pollution/forecast", {"lat": lat, "lon": lon})
        return [self._parse_item(item) for item in data.get("list", [])]

    def history(self, lat: float, lon: float,
                start: int, end: int) -> list[dict]:
        """Historical air quality between UNIX timestamps."""
        data = self._get("air_pollution/history",
                         {"lat": lat, "lon": lon, "start": start, "end": end})
        return [self._parse_item(item) for item in data.get("list", [])]

    @staticmethod
    def _parse_item(item: dict) -> dict:
        comp = item.get("components", {})
        return {
            "timestamp":  datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M"),
            "aqi_owm":    item["main"]["aqi"],       # 1=Good … 5=Very Poor
            "co":         comp.get("co",    None),   # μg/m³
            "no":         comp.get("no",    None),
            "no2":        comp.get("no2",   None),
            "o3":         comp.get("o3",    None),
            "so2":        comp.get("so2",   None),
            "nh3":        comp.get("nh3",   None),
            "pm2_5":      comp.get("pm2_5", None),
            "pm10":       comp.get("pm10",  None),
        }

    @staticmethod
    def _parse(data: dict) -> dict:
        return AirPollutionAPI._parse_item(data["list"][0])

    @staticmethod
    def owm_to_india_aqi(owm_aqi: int, pm2_5: float, pm10: float) -> float:
        """
        Convert OWM 1-5 index + PM data to an approximate India CPCB AQI.
        Uses PM2.5 sub-index as primary driver (most influential in India).
        """
        # PM2.5 breakpoints (μg/m³) → India AQI sub-index
        pm25_bp = [(0, 30, 0, 50), (30, 60, 51, 100),
                   (60, 90, 101, 200), (90, 120, 201, 300),
                   (120, 250, 301, 400), (250, 380, 401, 500)]
        sub = 0.0
        for lo_c, hi_c, lo_i, hi_i in pm25_bp:
            if lo_c <= pm2_5 <= hi_c:
                sub = lo_i + (pm2_5 - lo_c) * (hi_i - lo_i) / (hi_c - lo_c)
                break
        return round(max(sub, owm_aqi * 80), 1)  # blend with scale fallback

    def demo_fetch(self, lat: float = 28.6139, lon: float = 77.2090):
        """Demo fetch for Delhi. Replace lat/lon for any city."""
        print(f"\n🌐 OpenWeatherMap API — current reading at ({lat}, {lon})")
        try:
            data = self.current(lat, lon)
            print(json.dumps(data, indent=2))

            forecast = self.forecast(lat, lon)
            # Daily aggregation (next 7 days)
            df_fc = pd.DataFrame(forecast)
            df_fc["date"] = pd.to_datetime(df_fc["timestamp"]).dt.date
            daily = df_fc.groupby("date").agg({
                "pm2_5": "mean", "pm10": "mean",
                "no2": "mean", "o3": "mean",
                "so2": "mean", "co": "mean",
                "nh3": "mean", "aqi_owm": "mean"
            }).round(2).head(7)
            print("\n📅 7-Day Forecast (daily avg):")
            print(daily.to_string())
            return daily
        except Exception as e:
            print(f"   ⚠️  API call failed: {e}")
            print("   (Using synthetic demo data instead)")
            return self._synthetic_forecast()

    @staticmethod
    def _synthetic_forecast() -> pd.DataFrame:
        """Synthetic data for offline testing."""
        dates = [datetime.today().date() + timedelta(days=i) for i in range(7)]
        np.random.seed(42)
        return pd.DataFrame({
            "date":    dates,
            "pm2_5":  np.random.uniform(20, 120, 7).round(1),
            "pm10":   np.random.uniform(40, 200, 7).round(1),
            "no2":    np.random.uniform(5,  60,  7).round(1),
            "o3":     np.random.uniform(20, 80,  7).round(1),
            "so2":    np.random.uniform(2,  30,  7).round(1),
            "co":     np.random.uniform(200, 800, 7).round(1),
            "nh3":    np.random.uniform(1,  20,  7).round(1),
            "aqi_owm": np.random.randint(1, 5, 7),
        }).set_index("date")


# ═════════════════════════════════════════════
# 3. MODEL TRAINING
# ═════════════════════════════════════════════
class AQIModelTrainer:
    """Train & evaluate multiple regressors; select the best."""

    MODELS = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42
        ),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
    }

    def __init__(self):
        self.best_model  = None
        self.best_name   = None
        self.results     = {}
        self.scaler      = StandardScaler()

    def train_evaluate(self, X: np.ndarray, y: np.ndarray):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_te_sc = self.scaler.transform(X_te)

        best_r2 = -np.inf
        print("\n" + "═" * 60)
        print("  MODEL TRAINING & EVALUATION")
        print("═" * 60)

        for name, model in self.MODELS.items():
            model.fit(X_tr_sc, y_tr)
            y_pred = model.predict(X_te_sc)

            mae  = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            mape = mean_absolute_percentage_error(y_te, y_pred) * 100

            # 5-fold CV on training set
            cv_scores = cross_val_score(
                model, X_tr_sc, y_tr, cv=5,
                scoring="neg_mean_absolute_error"
            )
            cv_mae = -cv_scores.mean()

            self.results[name] = dict(mae=mae, rmse=rmse, r2=r2,
                                      mape=mape, cv_mae=cv_mae)

            print(f"\n  [{name}]")
            print(f"    MAE  : {mae:.2f}  |  RMSE : {rmse:.2f}")
            print(f"    R²   : {r2:.4f}  |  MAPE : {mape:.1f}%")
            print(f"    CV-MAE (5-fold): {cv_mae:.2f}")

            if r2 > best_r2:
                best_r2          = r2
                self.best_model  = model
                self.best_name   = name

        print(f"\n✅ Best model: {self.best_name}  (R² = {best_r2:.4f})")
        return self.best_model

    def feature_importance(self, feature_cols: list):
        if hasattr(self.best_model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature":    feature_cols,
                "importance": self.best_model.feature_importances_
            }).sort_values("importance", ascending=False)
            print("\n📊 Top-10 Feature Importances:")
            print(fi.head(10).to_string(index=False))
            return fi
        return None

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.best_model, f"{model_dir}/best_model.pkl")
        joblib.dump(self.scaler,     f"{model_dir}/scaler.pkl")
        with open(f"{model_dir}/results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Artifacts saved to '{model_dir}/'")

    @staticmethod
    def load(model_dir: str):
        model  = joblib.load(f"{model_dir}/best_model.pkl")
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        return model, scaler


# ═════════════════════════════════════════════
# 4. 7-DAY PREDICTOR
# ═════════════════════════════════════════════
class AQIForecaster:
    """
    Generate 7-day AQI predictions for a given state/area.
    Optionally enriches features with live OpenWeatherMap data.
    """

    def __init__(self, model, scaler, processor: AQIDataProcessor,
                 feature_cols: list):
        self.model        = model
        self.scaler       = scaler
        self.processor    = processor
        self.feature_cols = feature_cols

    def predict_7days(
        self,
        df_full: pd.DataFrame,
        target_state: str,
        target_area:  str,
        api_client:   AirPollutionAPI = None,
        lat: float = None,
        lon: float = None,
    ) -> pd.DataFrame:

        # Latest row for the chosen area
        subset = df_full[
            (df_full["state"].str.lower() == target_state.lower()) &
            (df_full["area"].str.lower()  == target_area.lower())
        ].tail(30).copy()

        if subset.empty:
            raise ValueError(f"No data found for {target_area}, {target_state}.")

        last_row  = subset.iloc[-1]
        last_date = last_row["date"]
        last_aqi  = last_row["aqi_value"]

        # Optionally fetch API forecast
        api_daily = None
        if api_client and lat and lon:
            api_daily = api_client.demo_fetch(lat, lon)

        # ── Iterative 7-day rollout ────────────────────────────────
        predictions = []
        hist_aqis   = list(subset["aqi_value"].values)   # rolling history

        for d in range(1, PREDICTION_DAYS + 1):
            future_date = last_date + timedelta(days=d)

            lag1  = hist_aqis[-1]  if len(hist_aqis) >= 1  else last_aqi
            lag7  = hist_aqis[-7]  if len(hist_aqis) >= 7  else last_aqi
            lag30 = hist_aqis[-30] if len(hist_aqis) >= 30 else last_aqi
            roll7  = np.mean(hist_aqis[-7:])
            roll30 = np.mean(hist_aqis[-30:] if len(hist_aqis) >= 30 else hist_aqis)
            std7   = np.std(hist_aqis[-7:])
            trend  = (np.polyfit(range(min(7, len(hist_aqis))),
                                 hist_aqis[-7:], 1)[0]
                      if len(hist_aqis) >= 2 else 0.0)

            # Encode state/area safely
            try:
                state_enc = self.processor.le_state.transform([target_state.title()])[0]
            except ValueError:
                state_enc = 0
            try:
                area_enc = self.processor.le_area.transform([target_area.title()])[0]
            except ValueError:
                area_enc = 0
            try:
                poll_enc = self.processor.le_poll.transform(
                    [str(last_row.get("prominent_pollutants", "Pm10")).title()]
                )[0]
            except ValueError:
                poll_enc = 0

            features = [
                future_date.year,
                future_date.month,
                future_date.weekday(),
                future_date.timetuple().tm_yday,
                (future_date.month - 1) // 3 + 1,
                int(future_date.weekday() >= 5),
                {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[future_date.month],
                state_enc, area_enc, poll_enc,
                int(last_row.get("monitoring_stations",
                                 last_row.get("number_of_monitoring_stations", 1))),
                lag1, lag7, lag30,
                roll7, roll30, std7, trend
            ]

            X_pred    = self.scaler.transform([features])
            pred_aqi  = float(np.clip(self.model.predict(X_pred)[0], 0, 500))

            # If API data available, blend PM2.5-based AQI estimate
            if api_daily is not None and d <= len(api_daily):
                row_fc    = api_daily.iloc[min(d - 1, len(api_daily) - 1)]
                api_aqi   = AirPollutionAPI.owm_to_india_aqi(
                    int(row_fc["aqi_owm"]),
                    float(row_fc["pm2_5"]),
                    float(row_fc["pm10"])
                )
                pred_aqi  = round(0.65 * pred_aqi + 0.35 * api_aqi, 1)
                pm25, pm10 = round(float(row_fc["pm2_5"]), 1), round(float(row_fc["pm10"]), 1)
                no2, o3    = round(float(row_fc["no2"]), 1), round(float(row_fc["o3"]), 1)
                so2, co    = round(float(row_fc["so2"]), 1), round(float(row_fc["co"]), 1)
                nh3        = round(float(row_fc["nh3"]), 1)
            else:
                pm25=pm10=no2=o3=so2=co=nh3 = None

            hist_aqis.append(pred_aqi)

            predictions.append({
                "date":         future_date.strftime("%Y-%m-%d"),
                "day":          future_date.strftime("%A"),
                "predicted_aqi": round(pred_aqi, 1),
                "category":     aqi_category(pred_aqi),
                "pm2_5_μg/m³":  pm25,
                "pm10_μg/m³":   pm10,
                "no2_μg/m³":    no2,
                "o3_μg/m³":     o3,
                "so2_μg/m³":    so2,
                "co_μg/m³":     co,
                "nh3_μg/m³":    nh3,
            })

        result_df = pd.DataFrame(predictions)
        return result_df


# ═════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═════════════════════════════════════════════
def main():
    print("=" * 60)
    print("   AQI 7-DAY PREDICTION SYSTEM WITH OPENAPI FEATURES")
    print("=" * 60)

    # ── 5.1 Data preparation ──────────────────────────────────────
    processor = AQIDataProcessor(CSV_PATH)
    df_raw    = processor.load()
    df_clean  = processor.clean(df_raw)
    df_feat   = processor.feature_engineer(df_clean)

    feature_cols = processor.get_feature_cols()
    X = df_feat[feature_cols].values
    y = df_feat["aqi_value"].values

    # ── 5.2 Model training ────────────────────────────────────────
    trainer = AQIModelTrainer()
    trainer.train_evaluate(X, y)
    trainer.feature_importance(feature_cols)
    trainer.save(MODEL_DIR)

    # ── 5.3 OpenWeatherMap demo ───────────────────────────────────
    api = AirPollutionAPI(OPENWEATHER_API_KEY)

    # ── 5.4 7-day forecast ───────────────────────────────────────
    forecaster = AQIForecaster(
        trainer.best_model,
        trainer.scaler,
        processor,
        feature_cols
    )

    # Example: predict for Delhi, Delhi
    TARGET_STATE = "Delhi"
    TARGET_AREA  = "Delhi"
    LAT, LON     = 28.6139, 77.2090   # Delhi coordinates

    print("\n" + "=" * 60)
    print(f"  7-DAY AQI FORECAST — {TARGET_AREA.upper()}, {TARGET_STATE.upper()}")
    print("=" * 60)

    forecast_df = forecaster.predict_7days(
        df_feat,
        TARGET_STATE,
        TARGET_AREA,
        api_client=api,
        lat=LAT,
        lon=LON,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(forecast_df.to_string(index=False))

    # Save forecast
    out = "7day_aqi_forecast.csv"
    forecast_df.to_csv(out, index=False)
    print(f"\n✅ Forecast saved → {out}")

    # ── 5.5 Bulk forecast for all states ─────────────────────────
    print("\n" + "=" * 60)
    print("  BULK FORECAST — LATEST AQI SUMMARY BY STATE")
    print("=" * 60)
    latest = (
        df_feat.sort_values("date")
        .groupby("state")
        .last()
        .reset_index()[["state", "aqi_value", "prominent_pollutants"]]
    )
    latest["category"] = latest["aqi_value"].apply(aqi_category)
    latest.columns      = ["State", "Latest AQI", "Prominent Pollutant", "Category"]
    print(latest.sort_values("Latest AQI", ascending=False).to_string(index=False))

    print("\n🎉 Pipeline complete!\n")


if __name__ == "__main__":
    main()
