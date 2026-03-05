"""
this code Transcribe audio to timestamped segments.

Uses OpenAI Whisper (runs fully locally, no API call needed).
Model is loaded once and reused if called multiple times in a session.
"""

import os
from functools import lru_cache

import whisper

import sys
from pathlib import Path as _root_path
sys.path.insert(0, str(_root_path(__file__).resolve().parent.parent))

from models import TranscriptSegment


@lru_cache(maxsize=1)
def _load_model(model_size: str) -> whisper.Whisper:
    """Load and cache the Whisper model."""
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model_size: str | None = None) -> list[TranscriptSegment]:
    size = model_size or os.getenv("WHISPER_MODEL", "base")
    model = _load_model(size)

    # verbose=False suppresses per-segment print spam
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