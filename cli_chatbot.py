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
    
def choose_mode() -> str:
    print("How would you like to talk to GridWise?")
    print("  1) Text chat   (type your messages)")
    print("  2) Voice chat  (speak and listen)")
    while True:
        try:
            pick = input("Choose 1 or 2: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "text"
        if pick in {"1", "text", "t"}:
            return "text"
        if pick in {"2", "voice", "v"}:
            return "voice"
        print("Please type 1 or 2.")


def voice_ready() -> bool:
    """Check voice support and explain what is missing if it is not ready."""
    if voice.microphone_available():
        return True
    print("\n[voice] A microphone or the voice packages were not found.")
    for name, ok in voice.status().items():
        print(f"   {'OK     ' if ok else 'MISSING'}  {name}")
    print("   Install with:  pip install SpeechRecognition pyaudio pyttsx3")
    print("   Staying in text chat.\n")
    return False