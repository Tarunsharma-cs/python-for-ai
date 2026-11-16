"""Step 8 (optional): Streamlit web app with custom HTML + CSS.

Run from the project folder:  streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "final_metrics.json"
PARAMS_PATH = ROOT / "reports" / "best_params.json"

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="🚚", layout="centered")

# ---------- Inject the CSS file ----------
css = (APP_DIR / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML (kept free of leading spaces so Markdown does not treat it as code)."""
    st.markdown(markup, unsafe_allow_html=True)


# ---------- Hero banner ----------
html(
    '<div class="hero">'
    '<div class="badge">🚚 Transport &amp; Logistics · Machine Learning</div>'
    "<h1>Delivery Delay Predictor</h1>"
    "<p>Enter the delivery details and find out whether it is likely to arrive on time.</p>"
    "</div>"
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/run_all.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------- Sidebar: model facts ----------
with st.sidebar:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        model_name = "Logistic Regression"
        if PARAMS_PATH.exists():
            model_name = json.loads(PARAMS_PATH.read_text()).get("model", model_name)
        html(
            f'<div class="side-title">About the model</div>'
            f'<div class="side-sub">{model_name}, tested on unseen data</div>'
            '<div class="tile-grid">'
            f'<div class="tile"><div class="value">{metrics["accuracy"]:.1%}</div><div class="name">Accuracy</div></div>'
            f'<div class="tile"><div class="value">{metrics["precision"]:.1%}</div><div class="name">Precision</div></div>'
            f'<div class="tile"><div class="value">{metrics["recall"]:.1%}</div><div class="name">Recall</div></div>'
            f'<div class="tile"><div class="value">{metrics["roc_auc"]:.3f}</div><div class="name">ROC-AUC</div></div>'
            "</div>"
        )
    else:
        html('<div class="side-title">About the model</div>')

# ---------- Inputs ----------
html('<div class="section-title">Delivery details</div>')
col1, col2 = st.columns(2)
with col1:
    distance = st.number_input("📏 Distance (km)", min_value=0.5, max_value=100.0, value=12.0, step=0.5)
    vehicle = st.selectbox("🛵 Vehicle type", ["Bike", "Scooter", "Car", "Van"])
    traffic = st.selectbox("🚦 Traffic level", ["Low", "Medium", "High"])
with col2:
    weather = st.selectbox("🌦️ Weather condition", ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"])
    rating = st.slider("⭐ Driver rating", 1.0, 5.0, 4.3, 0.1)
    priority = st.selectbox("📦 Delivery priority", ["Low", "Standard", "Express"])

st.write("")
predict = st.button("Predict delivery status")


def risk_factors(distance, traffic, weather, rating, priority):
    """Plain-language chips explaining which inputs push the risk up or down."""
    chips = []
    chips.append(("Heavy traffic", "bad") if traffic == "High" else
                 ("Light traffic", "good") if traffic == "Low" else ("Moderate traffic", "good"))
    chips.append((f"{weather} weather", "bad") if weather in ("Rainy", "Foggy", "Stormy")
                 else (f"{weather} weather", "good"))
    chips.append((f"Long trip ({distance:g} km)", "bad") if distance > 25
                 else (f"Short trip ({distance:g} km)", "good"))
    chips.append((f"Low driver rating ({rating:.1f})", "bad") if rating < 3.8
                 else (f"Good driver rating ({rating:.1f})", "good"))
    if priority == "Express":
        chips.append(("Express: tight time promise", "bad"))
    return chips


# ---------- Prediction ----------
if predict:
    row = pd.DataFrame(
        [
            {
                "distance_km": distance,
                "driver_rating": rating,
                "vehicle_type": vehicle,
                "traffic_level": traffic,
                "weather_condition": weather,
                "delivery_priority": priority,
            }
        ]
    )
    prob = float(model.predict_proba(row)[0, 1])
    pct = round(prob * 100)
    delayed = prob >= 0.5

    css_class = "delayed" if delayed else "ontime"
    icon = "⚠️" if delayed else "✅"
    label = "Likely DELAYED" if delayed else "Likely ON TIME"
    chips = "".join(f'<span class="chip {kind}">{text}</span>' for text, kind in
                    risk_factors(distance, traffic, weather, rating, priority))

    html(
        f'<div class="result-card {css_class}">'
        f'<div class="result-icon">{icon}</div>'
        f'<div class="result-body"><div class="result-label">{label}</div>'
        '<div class="result-sub">Estimated probability of delay</div></div>'
        f'<div class="result-pct">{pct}%</div>'
        "</div>"
        '<div class="gauge"><div class="gauge-track">'
        f'<div class="gauge-marker" style="left:{pct}%"></div></div>'
        '<div class="gauge-scale"><span>On time</span><span>Uncertain</span><span>Delayed</span></div></div>'
        '<div class="section-title" style="margin-top:1.2rem">What affects this delivery</div>'
        f'<div class="chips">{chips}</div>'
    )

html('<div class="app-footer">Field Project · Transport &amp; Logistics · built with scikit-learn and Streamlit</div>')
