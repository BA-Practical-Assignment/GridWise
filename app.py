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