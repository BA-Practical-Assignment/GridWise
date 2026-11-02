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

def say(text: str, mode: str):
    print(f"Bot: {text}\n")
    if mode == "voice":
        voice.speak(text)


def get_user_input(mode: str):
    """Return (text, mode). In voice mode, falls back to typing if nothing is heard."""
    if mode == "voice":
        print("[listening... speak now]")
        heard = voice.listen()
        if heard:
            print(f"You (voice): {heard}")
            return heard, mode
        print("[I did not catch that - please type instead, or press Enter to try speaking again]")
        typed = input("You: ")
        return (typed if typed.strip() else ""), mode
    return input("You: "), mode


def chat(mode: str):
    bot = build_bot()
    if mode == "voice" and not voice_ready():
        mode = "text"
    print(f"\nGridWise assistant - {mode.upper()} mode "
          f"(say or type 'voice mode', 'text mode' or 'quit')\n")
    say(GREETING, mode)
    while True:
         try:
            msg, mode = get_user_input(mode)
        except (EOFError, KeyboardInterrupt):
            break
        low = msg.strip().lower()
        if not low:
            continue
        if low in QUIT_WORDS:
            break
        if low in TO_VOICE:
            if voice_ready():
                mode = "voice"
                say("Voice mode on. I'm listening.", mode)
            continue
        if low in TO_TEXT:
            mode = "text"
            say("Text mode on. Type your message.", mode)
            continue
        r = bot.reply(msg)
        say(r["reply"], mode)
        if r.get("intent") == "goodbye":
            break
    if bot.tickets:
        print("Tickets logged this session:")
        for t in bot.tickets:
            print(f"  {t['ticket']}  {t.get('category', t['intent'])}  urgency={t.get('urgency', '-')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chat with the GridWise assistant")
    ap.add_argument("--mode", choices=["text", "voice"], default=None,
                    help="start in text or voice mode (default: ask)")
    args = ap.parse_args()
    chat(args.mode or choose_mode())
