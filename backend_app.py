import os
import warnings
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta, date
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(title="AQI 7-Day Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROBUST PATH FINDER
# ─────────────────────────────────────────────
def find_file(filename):
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
        filename,
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

FILES = {
    "aqi.csv":                     find_file("aqi.csv"),
    "best_model.pkl":              find_file("best_model.pkl"),
    "scaler.pkl":                  find_file("scaler.pkl"),
    "label_encoder_state.pkl":     find_file("label_encoder_state.pkl"),
    "label_encoder_area.pkl":      find_file("label_encoder_area.pkl"),
    "label_encoder_pollutant.pkl": find_file("label_encoder_pollutant.pkl"),
}

OPENWEATHER_API_KEY = "YOUR_API_KEY"
PREDICTION_DAYS = 7

# Globals to hold loaded models
model = None
scaler = None
le_state = None
le_area = None
le_poll = None
df = None
state_area_map = {}

AQI_META = {
    "Good":         {"range": (0,   50),  "color": "#00c853"},
    "Satisfactory": {"range": (51,  100), "color": "#69b34c"},
    "Moderate":     {"range": (101, 200), "color": "#fab733"},
    "Poor":         {"range": (201, 300), "color": "#ff6600"},
    "Very Poor":    {"range": (301, 400), "color": "#e63946"},
    "Severe":       {"range": (401, 500), "color": "#7b2d8b"},
}

def aqi_category(val):
    for label, meta in AQI_META.items():
        lo, hi = meta["range"]
        if lo <= val <= hi:
            return label
    return "Severe"

def aqi_color(val):
    return AQI_META[aqi_category(val)]["color"]

def season_of(month):
    return {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}[month]

@app.on_event("startup")
def load_artifacts():
    global model, scaler, le_state, le_area, le_poll, df, state_area_map
    try:
        model    = joblib.load(FILES["best_model.pkl"])
        scaler   = joblib.load(FILES["scaler.pkl"])
        le_state = joblib.load(FILES["label_encoder_state.pkl"])
        le_area  = joblib.load(FILES["label_encoder_area.pkl"])
        le_poll  = joblib.load(FILES["label_encoder_pollutant.pkl"])
        
        df = pd.read_csv(FILES["aqi.csv"])
        df["state"] = df["state"].astype(str).str.strip().str.title()
        df["area"]  = df["area"].astype(str).str.strip().str.title()
        df["prominent_pollutants"] = df["prominent_pollutants"].astype(str).str.strip().str.title()
        df = df[(df["aqi_value"] >= 0) & (df["aqi_value"] <= 500)]
        df.dropna(subset=["aqi_value", "state", "area"], inplace=True)
        # Sort by area so the tail(60) grabs the most recent (assuming original CSV is chronological)
        df.sort_values(["area"], inplace=True)
        if "number_of_monitoring_stations" in df.columns:
            df["monitoring_stations"] = df["number_of_monitoring_stations"].fillna(1).astype(int)
        else:
            df["monitoring_stations"] = 1
            
        state_area_map = df.groupby("state")["area"].unique().apply(sorted).apply(list).to_dict()
    except Exception as e:
        print(f"Error loading artifacts: {e}")

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),      "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),  "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),  "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),     "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),      "Bhopal": (23.2599, 77.4126),
    "Surat": (21.1702, 72.8311),      "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),     "Chandigarh": (30.7333, 76.7794),
    "Visakhapatnam": (17.6868, 83.2185), "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),   "Guwahati": (26.1445, 91.7362),
}

def fetch_owm_forecast(lat, lon, api_key):
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
            params={"lat": lat, "lon": lon, "appid": api_key}, timeout=8
        )
        resp.raise_for_status()
        rows = []
        for item in resp.json().get("list", []):
            c = item.get("components", {})
            rows.append({
                "date":    datetime.utcfromtimestamp(item["dt"]).date(),
                "aqi_owm": item["main"]["aqi"],
                "co":    c.get("co", 0),    "no":  c.get("no",   0),
                "no2":   c.get("no2", 0),   "o3":  c.get("o3",   0),
                "so2":   c.get("so2", 0),   "nh3": c.get("nh3",  0),
                "pm2_5": c.get("pm2_5", 0), "pm10":c.get("pm10", 0),
            })
        return pd.DataFrame(rows).groupby("date").mean().round(2).head(7)
    except Exception:
        return None

def owm_to_india_aqi(owm_aqi, pm2_5):
    bp = [(0,30,0,50),(30,60,51,100),(60,90,101,200),
          (90,120,201,300),(120,250,301,400),(250,380,401,500)]
    sub = 0.0
    for lo_c, hi_c, lo_i, hi_i in bp:
        if lo_c <= pm2_5 <= hi_c:
            sub = lo_i + (pm2_5-lo_c)*(hi_i-lo_i)/(hi_c-lo_c)
            break
    return round(max(sub, owm_aqi * 80), 1)

def predict_7days_logic(state, area, api_key):
    subset = df[(df["state"] == state) & (df["area"] == area)].tail(60)
    if subset.empty:
        return None, f"No historical data found for {area}, {state}."

    last_row   = subset.iloc[-1]
    hist_aqis  = list(subset["aqi_value"].values)

    today      = datetime.combine(date.today(), datetime.min.time())
    start_date = today

    try:    st_enc = int(le_state.transform([state])[0])
    except: st_enc = 0
    try:    ar_enc = int(le_area.transform([area])[0])
    except: ar_enc = 0
    try:
        pv     = str(last_row.get("prominent_pollutants", "Pm10")).strip().title()
        po_enc = int(le_poll.transform([pv])[0])
    except: po_enc = 0

    monitoring = int(last_row.get("monitoring_stations", 1))

    api_daily = None
    if api_key and api_key != "YOUR_API_KEY":
        coords = CITY_COORDS.get(area)
        if coords:
            api_daily = fetch_owm_forecast(*coords, api_key)

    results = []
    for d in range(0, PREDICTION_DAYS):
        fd = start_date + timedelta(days=d)

        lag1  = float(hist_aqis[-1])
        lag7  = float(hist_aqis[-7])  if len(hist_aqis) >= 7  else float(hist_aqis[0])
        lag30 = float(hist_aqis[-30]) if len(hist_aqis) >= 30 else float(hist_aqis[0])
        roll7 = float(np.mean(hist_aqis[-7:]))
        roll30= float(np.mean(hist_aqis[-30:] if len(hist_aqis) >= 30 else hist_aqis))
        std7  = float(np.std(hist_aqis[-7:])) if len(hist_aqis) >= 2 else 0.0

        feat_vec = [
            fd.year, fd.month, fd.weekday(), fd.timetuple().tm_yday,
            (fd.month-1)//3+1, int(fd.weekday()>=5), season_of(fd.month),
            st_enc, ar_enc, po_enc, monitoring,
            lag1, lag7, lag30, roll7, roll30, std7,
        ]

        X_sc     = scaler.transform([feat_vec])
        pred_aqi = float(np.clip(model.predict(X_sc)[0], 0, 500))

        pm25=pm10=no2=o3=so2=co=nh3=no = None
        if api_daily is not None and d < len(api_daily):
            row_fc   = api_daily.iloc[min(d, len(api_daily)-1)]
            api_aqi  = owm_to_india_aqi(int(row_fc["aqi_owm"]), float(row_fc["pm2_5"]))
            pred_aqi = 0.65 * pred_aqi + 0.35 * api_aqi
            pm25=round(float(row_fc["pm2_5"]),1); pm10=round(float(row_fc["pm10"]),1)
            no2 =round(float(row_fc["no2"]),1);   o3  =round(float(row_fc["o3"]),1)
            so2 =round(float(row_fc["so2"]),1);   co  =round(float(row_fc["co"]),1)
            nh3 =round(float(row_fc["nh3"]),1);   no  =round(float(row_fc["no"]),1)
        else:
            # Fallback to estimate if API fails or no API KEY
            pm25 = round(pred_aqi * 0.45, 1)
            pm10 = round(pred_aqi * 0.7, 1)
            no2 = round(pred_aqi * 0.15, 1)
            o3 = round(pred_aqi * 0.2, 1)
            so2 = round(pred_aqi * 0.1, 1)
            co = round(pred_aqi * 0.05, 1)
            nh3 = round(pred_aqi * 0.02, 1)
            no = round(pred_aqi * 0.08, 1)

        pred_aqi = round(pred_aqi, 1)
        hist_aqis.append(pred_aqi)
        cat = aqi_category(pred_aqi)

        if d == 0:
            day_label = "TODAY"
        elif d == 1:
            day_label = "TOMORROW"
        else:
            day_label = fd.strftime("%a").upper()

        meta = AQI_META[cat]
        meta_with_label = {"label": cat, "color": meta["color"], "class": cat.lower().replace(' ', '-')}

        results.append({
            "dateObj": fd.isoformat(),
            "dateStr": fd.strftime("%d %b %Y"),
            "day": day_label,
            "aqi": pred_aqi,
            "meta": meta_with_label,
            "pm25": pm25, "pm10": pm10,
            "no2": no2, "o3": o3,
            "so2": so2, "co": co,
            "nh3": nh3, "no": no
        })
        
    return results, None

class PredictionRequest(BaseModel):
    state: str
    city: str

@app.get("/api/locations")
def get_locations():
    return state_area_map

@app.post("/api/predict")
def predict_aqi(req: PredictionRequest):
    if df is None or model is None:
        raise HTTPException(status_code=500, detail="Models or Data not loaded correctly.")
        
    results, err = predict_7days_logic(req.state, req.city, OPENWEATHER_API_KEY)
    
    if err:
        raise HTTPException(status_code=404, detail=err)
        
    return {"forecast": results}

# Mount the static frontend
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
