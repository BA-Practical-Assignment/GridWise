from __future__ import annotations

import io
import os
import re
import tempfile

LANGUAGE = "en-ZA"          
def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def stt_available() -> bool:
    """Can we turn audio (a WAV file/bytes) into text?"""
    return _has("speech_recognition")


def microphone_available() -> bool:
    """Can the TERMINAL chat record from a microphone? (needs PyAudio)"""
    if not stt_available() or not _has("pyaudio"):
        return False
    try:
        import speech_recognition as sr
        return len(sr.Microphone.list_microphone_names()) > 0
    except Exception:
        return False


def tts_available() -> bool:
    return _has("pyttsx3") or _has("gtts")


def status() -> dict:
    """Short report used by the CLI and the dashboard to explain what is missing."""
    return {
        "speech_to_text": stt_available(),
        "microphone (terminal)": microphone_available(),
        "text_to_speech_offline (pyttsx3)": _has("pyttsx3"),
        "text_to_speech_online (gTTS)": _has("gtts"),
    }



def clean_for_speech(text: str) -> str:
    """Tidy a chatbot reply so a speech engine reads it well."""
    
    text = re.sub(r"\bGW-([A-Z0-9]{6})\b",
                  lambda m: "G W " + " ".join(m.group(1)), text)
    text = text.replace("MW", "megawatts").replace("&", " and ")
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\[insert[^\]]*\]", "the number on our website", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()



_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 175)
    return _engine