import json

import joblib
import pandas as pd
import streamlit as st
import config
import fault_classifier as fcls
import forecasting as fc
import voice
from chatbot import GridWiseChatbot

st.set_page_config(page_title="GridWise", page_icon="⚡", layout="wide")

@st.cache_resource
def load_assets():
    bundle = joblib.load(config.MODELS_DIR / "forecaster.joblib")
    fault_models = fcls.load()
    bot = GridWiseChatbot(fault_models=fault_models, forecast_fn=fc.make_forecast_fn(bundle)).train()
    metrics = json.load(open(config.RESULTS_DIR / "metrics.json"))
    return bundle, fault_models, bot, metrics

try:
    bundle, fault_models, bot, metrics = load_assets()
except FileNotFoundError:
    st.error("Models not found. Run `python run_pipeline.py` first.")
    st.stop()

st.title("⚡ GridWise")
st.caption("AI-powered electricity demand forecasting and customer support")
if metrics["data_source"] == "synthetic":
    st.warning("Running on SYNTHETIC demonstration data. Re-run the pipeline on the real PJM data "
               "for the final results.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand forecast", "🛠️ Fault classifier", "💬 Chatbot", "📊 Model results"])

with tab1:
    forecast = fc.forecast_next_day(bundle["model"], bundle["history"])
    thr = bundle["threshold"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak forecast (MW)", f"{forecast.max():,.0f}")
    c2.metric("Peak hour", f"{forecast.idxmax():%H:%M}")
    c3.metric("Hours above capacity warning", int((forecast > thr).sum()))
    chart = pd.DataFrame({"forecast_MW": forecast, "capacity_warning_MW": thr})
    st.line_chart(chart)
    msg = fc.describe_forecast(forecast, thr)
    (st.error if (forecast > thr).any() else st.success)(msg)
