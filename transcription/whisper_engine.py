import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import base64

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

    client   = get_client()
    response = client.models.generate_content(
        model    = _model(),
        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            types.Part.from_text(
                "Transcribe this audio completely and accurately. "
                "Format as timestamped segments, one per line:\n"
                "[MM:SS] transcript text here\n\n"
                "Include all spoken content. "
                "If exact timestamps are unclear, space them ~30 seconds apart."
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    return _parse_gemini_transcript(response.text or "")


def _parse_gemini_transcript(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    pattern = re.compile(r"^\[(\d{1,2}):(\d{2})\]\s*(.+)$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            m, s, content = match.groups()
            start = float(int(m) * 60 + int(s))
            # Close previous segment
            if segments:
                prev = segments[-1]
                segments[-1] = TranscriptSegment(
                    start=prev.start, end=start, text=prev.text
                )
            segments.append(TranscriptSegment(
                start=start, end=start + 30.0, text=content.strip()
            ))
        elif segments and line:
            # Continuation — append to last segment
            last = segments[-1]
            segments[-1] = TranscriptSegment(
                start=last.start, end=last.end,
                text=last.text + " " + line
            )

    if not segments and text.strip():
        segments.append(TranscriptSegment(start=0.0, end=30.0, text=text.strip()))

    return segments


def _transcribe_whisper(audio_path: str, model_size: str) -> list[TranscriptSegment]:
    """Local Whisper transcription — requires openai-whisper + torch installed."""
    from functools import lru_cache
    import whisper

    @lru_cache(maxsize=1)
    def _load(size: str):
        return whisper.load_model(size)

    result   = _load(model_size).transcribe(audio_path, verbose=False, task="transcribe")
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append(TranscriptSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
            ))
    return segments


def transcribe(
    audio_path: str,
    model_size: str | None = None,
) -> list[TranscriptSegment]:

    use_whisper = os.getenv("USE_WHISPER", "false").lower() == "true"

    if use_whisper:
        size = model_size or get_whisper_model()
        return _transcribe_whisper(audio_path, size)

    return _transcribe_gemini(audio_path)