import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:

    val = os.getenv(key, "")
    if val:
        return val

    try:
        import streamlit as st
        val = st.secrets.get(key, default)
        return str(val) if val else default
    except Exception:
        return default


def get_gemini_api_key() -> str:
    key = _get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "  Local:            add to .env file\n"
            "  Streamlit Cloud:  add in App Settings → Secrets\n"
            "  Get a free key:   https://aistudio.google.com"
        )
    return key


def get_gemini_model() -> str:
    return _get("GEMINI_MODEL", "gemini-2.5-flash")


def get_whisper_model() -> str:
    """Whisper model size, only used when running locally."""
    return _get("WHISPER_MODEL", "base")


def get_ffmpeg_path() -> str:
    return _get("FFMPEG_PATH", "")


def has_gemini_key() -> bool:
    try:
        get_gemini_api_key()
        return True
    except EnvironmentError:
        return False


def is_cloud() -> bool:
    """
    Detect if running on Streamlit Cloud.
    Used to skip Whisper (too heavy) and use Gemini transcription instead.
    """
    # Streamlit Cloud sets this env var on all hosted apps
    return os.getenv("STREAMLIT_SHARING_MODE", "") != "" or \
           os.getenv("IS_STREAMLIT_CLOUD", "") != ""