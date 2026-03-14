import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
from dotenv import load_dotenv

load_dotenv()

 
def get_gemini_api_key() -> str:
    """Return Gemini API key, raising clearly if not set."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com\n"
            "Then add to .env: GEMINI_API_KEY=your_key_here"
        )
    return key


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_whisper_model() -> str:
    return os.getenv("WHISPER_MODEL", "base")


def get_ffmpeg_path() -> str:
    return os.getenv("FFMPEG_PATH", "")


def has_gemini_key() -> bool:
    """Non-raising check - used by UI to show config error gracefully."""
    return bool(os.getenv("GEMINI_API_KEY", ""))