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

with tab2:
    st.write("Paste a customer fault report and GridWise will classify its category and urgency.")
    text = st.text_area("Fault report", "A cable is sparking on the ground near Main Road")
    if st.button("Classify"):
        res = fcls.classify(fault_models, text)
        st.write(f"**Category:** {res['category']}  (confidence {res['category_confidence']:.0%})")
        st.write(f"**Urgency:** {res['urgency']}")

with tab3:
    
    mode = st.radio("How would you like to chat?", ["💬 Text chat", "🎙️ Voice chat"], horizontal=True)
    voice_mode = mode.startswith("🎙️")
    if voice_mode:
        speak_replies = True
        if not voice.stt_available():
            st.warning("Voice input needs the SpeechRecognition package:  `pip install SpeechRecognition`. "
                       "Until then you can still type below.")
        if not voice.tts_available():
            st.info("To HEAR the replies install a speech engine:  `pip install pyttsx3` (offline) "
                    "or `pip install gTTS` (online). Replies will show as text until then.")
    else:
        speak_replies = st.toggle("🔊 Read replies aloud", value=False,
                                  help="Needs pyttsx3 or gTTS (see voice.py)")

    if "history" not in st.session_state:
        st.session_state.history = [("bot", "Hello! I'm the GridWise assistant. How can I help?")]
    st.session_state.setdefault("mic_n", 0)

    def handle_message(text: str, speak: bool):
        """Send one user message to the bot and (optionally) prepare a spoken reply."""
        st.session_state.history.append(("user", text))
        reply = bot.reply(text)["reply"]
        st.session_state.history.append(("bot", reply))
        if speak:
            audio_bytes, mime = voice.synthesize(reply)
            if audio_bytes:
                st.session_state.reply_audio = (audio_bytes, mime)

    
    for who, msg in st.session_state.history:
        with st.chat_message("assistant" if who == "bot" else "user"):
            st.write(msg)
    reply_audio = st.session_state.pop("reply_audio", None)      
    if reply_audio:
        st.audio(reply_audio[0], format=reply_audio[1], autoplay=True)

    
    if voice_mode:
        mic_key = f"mic_{st.session_state.mic_n}"
        if hasattr(st, "audio_input"):
            clip = st.audio_input("Press the microphone, speak, then press stop", key=mic_key)
        else:   
            clip = st.file_uploader("Upload a WAV recording of your question", type=["wav"], key=mic_key)
        if clip is not None:
            with st.spinner("Listening..."):
                heard = voice.transcribe(clip.getvalue())
            st.session_state.mic_n += 1                  
            if heard:
                handle_message(heard, speak=True)
            else:
                st.session_state.history.append(
                    ("bot", "Sorry, I could not hear that clearly. Please try again or type your message."))
            st.rerun()

    
    prompt = st.chat_input("Type here: outages, billing, faults, tips or the demand forecast")
    if prompt:
        handle_message(prompt, speak=speak_replies)
        st.rerun()

    if bot.tickets:
        with st.expander("Logged tickets"):
            st.dataframe(pd.DataFrame(bot.tickets))

with tab4:
    st.subheader("Forecast accuracy on the test period")
    st.dataframe(pd.DataFrame(metrics["forecast_metrics"]).T.round(2))
    for img in ["05_forecast_vs_actual.png", "06_model_comparison.png", "02_decomposition.png", "03_acf.png"]:
        p = config.RESULTS_DIR / img
        if p.exists():
            st.image(str(p))
