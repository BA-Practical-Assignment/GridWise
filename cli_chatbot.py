import argparse

import joblib

import config
import fault_classifier as fcls
import forecasting as fc
import voice
from chatbot import GridWiseChatbot

GREETING = "Hello! I'm the GridWise assistant. How can I help?"
QUIT_WORDS = {"quit", "exit", "stop", "close"}
TO_VOICE = {"voice", "voice mode", "switch to voice", "/voice"}
TO_TEXT = {"text", "text mode", "switch to text", "/text"}


def build_bot() -> GridWiseChatbot:
    fault_path = config.MODELS_DIR / "fault_classifier.joblib"
    bundle_path = config.MODELS_DIR / "forecaster.joblib"
    fault_models = fcls.load() if fault_path.exists() else None
    forecast_fn = fc.make_forecast_fn(joblib.load(bundle_path)) if bundle_path.exists() else None
    return GridWiseChatbot(fault_models=fault_models, forecast_fn=forecast_fn).train()