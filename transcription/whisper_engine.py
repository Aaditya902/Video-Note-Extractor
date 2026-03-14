"""
this code Transcribe audio to timestamped segments.

Uses OpenAI Whisper (runs fully locally, no API call needed).
Model is loaded once and reused if called multiple times in a session.
"""
import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    
from functools import lru_cache

import whisper
from config import get_whisper_model
from models import TranscriptSegment


    
@lru_cache(maxsize=1)
def _load_model(model_size: str) -> whisper.Whisper:
    """Load and cache the Whisper model."""
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model_size: str | None = None) -> list[TranscriptSegment]:
    size = model_size or get_whisper_model()
    model = _load_model(size)


    result = model.transcribe(audio_path, verbose=False, task="transcribe")

    segments: list[TranscriptSegment] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=text,
                )
            )

    return segments