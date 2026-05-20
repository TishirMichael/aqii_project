"""
AQI 7-Day Prediction System — Streamlit App
Predictions always start from TODAY's date.
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# ROBUST PATH FINDER
# ─────────────────────────────────────────────
def find_file(filename):
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
        os.path.join("/app", filename),
        os.path.join("/mount/src", filename),
        filename,
    ]
    for root, dirs, files in os.walk(os.getcwd()):
        if filename in files:
            search_paths.append(os.path.join(root, filename))
        if root.count(os.sep) - os.getcwd().count(os.sep) > 3:
            break
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

# EXACT 17 features — must match training
FEATURES = [
    'year', 'month', 'day_of_week', 'day_of_year', 'quarter',
    'is_weekend', 'season',
    'state_enc', 'area_enc', 'poll_enc', 'monitoring_stations',
    'aqi_lag_1', 'aqi_lag_7', 'aqi_lag_30',
    'aqi_roll_7d', 'aqi_roll_30d', 'aqi_std_7d'
]

OPENWEATHER_API_KEY = "YOUR_API_KEY"
PREDICTION_DAYS = 7

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="AQI 7-Day Prediction", page_icon="🌫️", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)

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

# ─────────────────────────────────────────────
# PAGE TITLE
# ─────────────────────────────────────────────
st.title("🌫️ AQI 7-Day Prediction System")
st.caption("India Air Quality Index Forecast · CO · NO · NO₂ · O₃ · SO₂ · NH₃ · PM2.5 · PM10")

# ─────────────────────────────────────────────
# CHECK FILES
# ─────────────────────────────────────────────
missing = [f for f, path in FILES.items() if path is None]
if missing:
    st.error("❌ Missing files!")
    for f in missing:
        st.markdown(f"- ❌ `{f}`")
    with st.expander("🔍 Debug Info"):
        st.write("**CWD:**", os.getcwd())
        st.write("**Files found:**", {k: v for k, v in FILES.items() if v is not None})
        try:
            all_files = []
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    all_files.append(os.path.join(root, file))
                if root.count(os.sep) - os.getcwd().count(os.sep) > 2:
                    break
            st.write("**All files visible:**", all_files)
        except Exception as e:
            st.write(str(e))
    st.markdown("""
    ### ✅ Fix: All files must be in ROOT of your GitHub repo:
    ```
    your_repo/
    ├── app.py
    ├── requirements.txt
    ├── aqi.csv
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoder_state.pkl
    ├── label_encoder_area.pkl
    └── label_encoder_pollutant.pkl
    ```
    """)
    st.stop()

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(FILES["best_model.pkl"])
    scaler   = joblib.load(FILES["scaler.pkl"])
    le_state = joblib.load(FILES["label_encoder_state.pkl"])
    le_area  = joblib.load(FILES["label_encoder_area.pkl"])
    le_poll  = joblib.load(FILES["label_encoder_pollutant.pkl"])
    return model, scaler, le_state, le_area, le_poll

@st.cache_data
def load_data():
    df = pd.read_csv(FILES["aqi.csv"])
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["state"] = df["state"].astype(str).str.strip().str.title()
    df["area"]  = df["area"].astype(str).str.strip().str.title()
    df["prominent_pollutants"] = df["prominent_pollutants"].astype(str).str.strip().str.title()
    df = df[(df["aqi_value"] >= 0) & (df["aqi_value"] <= 500)]
    df.dropna(subset=["date", "aqi_value", "state", "area"], inplace=True)
    df.sort_values(["area", "date"], inplace=True)
    if "number_of_monitoring_stations" in df.columns:
        df["monitoring_stations"] = df["number_of_monitoring_stations"].fillna(1).astype(int)
    else:
        df["monitoring_stations"] = 1
    return df

@st.cache_data
def build_state_area_map(_df):
    return _df.groupby("state")["area"].unique().apply(sorted).apply(list).to_dict()

# ─────────────────────────────────────────────
# OPENWEATHERMAP
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# PREDICTION — starts from TODAY
# ─────────────────────────────────────────────
def predict_7days(df, model, scaler, le_state, le_area, le_poll,
                  state, area, api_key):

    subset = df[(df["state"] == state) & (df["area"] == area)].tail(60)
    if subset.empty:
        return None, f"No historical data found for {area}, {state}."

    last_row   = subset.iloc[-1]
    hist_aqis  = list(subset["aqi_value"].values)

    # ── START FROM TODAY instead of last CSV date ──────────────
    today      = datetime.combine(date.today(), datetime.min.time())
    start_date = today  # Day 1 = today, Day 7 = today+6

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
    for d in range(0, PREDICTION_DAYS):   # 0 = today, 6 = today+6
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

        pred_aqi = round(pred_aqi, 1)
        hist_aqis.append(pred_aqi)
        cat = aqi_category(pred_aqi)

        # Label today specially
        if d == 0:
            day_label = "Today"
        elif d == 1:
            day_label = "Tomorrow"
        else:
            day_label = fd.strftime("%A")

        results.append({
            "Date":     fd.strftime("%d %b %Y"),
            "Day":      day_label,
            "AQI":      pred_aqi,
            "Category": cat,
            "Color":    aqi_color(pred_aqi),
            "PM2.5": pm25, "PM10": pm10,
            "NO2": no2,    "O3":   o3,
            "SO2": so2,    "CO":   co,
            "NH3": nh3,    "NO":   no,
        })

    return results, None

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
model, scaler, le_state, le_area, le_poll = load_artifacts()
df             = load_data()
state_area_map = build_state_area_map(df)

# Show today's date prominently
st.info(f"📅 Today is **{date.today().strftime('%d %B %Y')}** — Forecast will cover **{date.today().strftime('%d %b')} → {(date.today() + timedelta(days=6)).strftime('%d %b %Y')}**")

with st.expander("ℹ️ Model Info", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",         "GradientBoosting")
    c2.metric("R² Score",      "0.8063")
    c3.metric("MAE",           "20.61 AQI pts")
    c4.metric("Training Rows", f"{len(df):,}")

st.divider()

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    state = st.selectbox("📍 Select State", [""] + sorted(state_area_map.keys()))
with col2:
    areas = state_area_map.get(state, []) if state else []
    area  = st.selectbox("🏙️ Select Area / City", [""] + areas)
with col3:
    st.write(""); st.write("")
    predict_btn = st.button("🔍 Predict 7 Days", use_container_width=True, type="primary")

# Legend
lcols = st.columns(6)
for i, (cat, meta) in enumerate(AQI_META.items()):
    with lcols[i]:
        st.markdown(
            f"<div style='background:{meta['color']};border-radius:6px;padding:5px 8px;"
            f"text-align:center;font-size:0.72rem;font-weight:600;color:#fff'>"
            f"{cat}<br>{meta['range'][0]}–{meta['range'][1]}</div>",
            unsafe_allow_html=True
        )

st.write("")

if predict_btn:
    if not state or not area:
        st.warning("⚠️ Please select both State and Area.")
    else:
        with st.spinner(f"Predicting AQI for {area}, {state} …"):
            forecast, err = predict_7days(
                df, model, scaler, le_state, le_area, le_poll,
                state, area, OPENWEATHER_API_KEY
            )
        if err:
            st.error(err)
        else:
            st.success(f"✅ 7-Day Forecast for **{area}, {state}** — {date.today().strftime('%d %b')} to {(date.today()+timedelta(days=6)).strftime('%d %b %Y')}")

            # Forecast Cards
            st.subheader("📅 Daily AQI Forecast")
            ccols = st.columns(7)
            for i, day in enumerate(forecast):
                with ccols[i]:
                    st.markdown(f"""
                    <div style="background:{day['Color']}22;border:2px solid {day['Color']};
                         border-radius:12px;padding:14px 6px;text-align:center;margin-bottom:8px">
                      <div style="font-size:0.72rem;font-weight:700;color:{day['Color']}">{day['Day'].upper() if day['Day'] in ['Today','Tomorrow'] else day['Day'][:3].upper()}</div>
                      <div style="font-size:0.63rem;color:{day['Color']};opacity:0.85">{day['Date']}</div>
                      <div style="font-size:2rem;font-weight:800;color:{day['Color']};margin:6px 0">{day['AQI']}</div>
                      <div style="font-size:0.66rem;font-weight:600;color:{day['Color']}">{day['Category']}</div>
                    </div>""", unsafe_allow_html=True)

            # Bar Chart
            st.subheader("📊 AQI Trend")
            chart_df = pd.DataFrame({
                "Date": [d["Date"] for d in forecast],
                "AQI":  [d["AQI"]  for d in forecast],
            }).set_index("Date")
            st.bar_chart(chart_df, color="#3b82f6", height=220)

            # Metrics
            aqis = [d["AQI"] for d in forecast]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📈 Max AQI",        max(aqis))
            m2.metric("📉 Min AQI",        min(aqis))
            m3.metric("📊 Avg AQI",        round(np.mean(aqis), 1))
            m4.metric("🏷️ Worst Category", aqi_category(max(aqis)))

            # Table
            st.subheader("🧪 Full Details")
            has_poll = any(d["PM2.5"] is not None for d in forecast)
            if has_poll:
                st.dataframe(pd.DataFrame([{
                    "Date": d["Date"], "Day": d["Day"],
                    "AQI": d["AQI"], "Category": d["Category"],
                    "PM2.5 μg/m³": d["PM2.5"], "PM10 μg/m³": d["PM10"],
                    "NO₂ μg/m³": d["NO2"], "O₃ μg/m³": d["O3"],
                    "SO₂ μg/m³": d["SO2"], "CO μg/m³": d["CO"],
                    "NH₃ μg/m³": d["NH3"],
                } for d in forecast]), use_container_width=True, hide_index=True)
            else:
                st.dataframe(pd.DataFrame([{
                    "Date": d["Date"], "Day": d["Day"],
                    "Predicted AQI": d["AQI"], "Category": d["Category"],
                } for d in forecast]), use_container_width=True, hide_index=True)
                st.info("💡 Add OpenWeatherMap API key in app.py to see pollutant breakdown.")

            # Download
            dl_df = pd.DataFrame(forecast).drop(columns=["Color"])
            st.download_button(
                "⬇️ Download Forecast CSV",
                dl_df.to_csv(index=False),
                file_name=f"aqi_{area.replace(' ','_')}_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# Sidebar
with st.sidebar:
    st.header("📈 Historical AQI")
    if state and area:
        hist = df[(df["state"] == state) & (df["area"] == area)].tail(30)
        if not hist.empty:
            st.line_chart(
                hist.set_index("date")[["aqi_value"]].rename(columns={"aqi_value": "AQI"}),
                height=180
            )
            st.caption(f"Last 30 readings — {area}")
            latest = int(hist["aqi_value"].iloc[-1])
            delta  = int(hist["aqi_value"].iloc[-1] - hist["aqi_value"].iloc[-2]) if len(hist) >= 2 else None
            st.metric("Latest AQI", latest, delta=delta)
        else:
            st.info("No history available.")
    else:
        st.info("Select state & area to see history.")

    st.divider()
    st.markdown("**AQI Scale (India CPCB)**")
    for cat, meta in AQI_META.items():
        st.markdown(
            f"<div style='background:{meta['color']};color:#fff;padding:3px 10px;"
            f"border-radius:4px;font-size:0.78rem;margin-bottom:4px'>"
            f"{cat}: {meta['range'][0]}–{meta['range'][1]}</div>",
            unsafe_allow_html=True
        )
