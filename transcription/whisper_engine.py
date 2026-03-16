import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re

from config import get_whisper_model
from models import TranscriptSegment


def _transcribe_gemini(audio_path: str) -> list[TranscriptSegment]:
    """Transcribe audio using Gemini's native audio understanding."""
    from llm.gemini_client import get_client, get_gemini_model as _model
    from google.genai import types

    audio_bytes = Path(audio_path).read_bytes()
    ext         = Path(audio_path).suffix.lower()
    mime_map    = {
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
        ".m4a":  "audio/mp4",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    mime_type = mime_map.get(ext, "audio/mpeg")

    prompt = (
        "You are a transcription assistant. Transcribe the audio file completely.\n\n"
        "IMPORTANT — output format rules:\n"
        "- Output ONLY timestamped lines, nothing else\n"
        "- Every line MUST start with a timestamp in EXACTLY this format: [MM:SS]\n"
        "- Example: [00:00] Hello and welcome to this video.\n"
        "- Example: [01:30] Today we will discuss machine learning.\n"
        "- Example: [03:45] Let me show you the results.\n"
        "- Put a new timestamp every 20-30 seconds of audio\n"
        "- DO NOT write any intro, explanation, or summary — only the timestamped lines\n"
        "- If the audio is silent or unclear, write: [00:00] No speech detected."
    )

    client   = get_client()
    response = client.models.generate_content(
        model    = _model(),
        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )

    raw = response.text or ""
    segments = _parse_transcript(raw)

    # If parsing failed, return the whole text as one segment
    if not segments and raw.strip():
        segments = [TranscriptSegment(start=0.0, end=30.0, text=raw.strip())]

    return segments


def _parse_transcript(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []

    # Match [MM:SS] or [H:MM:SS] at start of line
    pattern = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.+)$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            a, b, c, content = match.groups()
            if c is not None:
                # [H:MM:SS]
                start = int(a) * 3600 + int(b) * 60 + int(c)
            else:
                # [MM:SS]
                start = int(a) * 60 + int(b)

            # Close previous segment
            if segments:
                prev = segments[-1]
                segments[-1] = TranscriptSegment(
                    start=prev.start,
                    end=float(start),
                    text=prev.text,
                )
            segments.append(TranscriptSegment(
                start=float(start),
                end=float(start) + 30.0,
                text=content.strip(),
            ))
        elif segments:
            # Continuation line — append to last segment
            last = segments[-1]
            segments[-1] = TranscriptSegment(
                start=last.start, end=last.end,
                text=last.text + " " + line,
            )

    return segments

def _transcribe_whisper(audio_path: str, model_size: str) -> list[TranscriptSegment]:
    from functools import lru_cache
    import whisper

    @lru_cache(maxsize=1)
    def _load(size: str):
        return whisper.load_model(size)

    result = _load(model_size).transcribe(audio_path, verbose=False, task="transcribe")
    return [
        TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
        )
        for seg in result.get("segments", [])
        if seg.get("text", "").strip()
    ]


def transcribe(
    audio_path: str,
    model_size: str | None = None,
) -> list[TranscriptSegment]:

    if os.getenv("USE_WHISPER", "false").lower() == "true":
        return _transcribe_whisper(audio_path, model_size or get_whisper_model())
    return _transcribe_gemini(audio_path)