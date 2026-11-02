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