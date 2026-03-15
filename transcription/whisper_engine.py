import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import get_whisper_model, is_cloud
from models import TranscriptSegment


def _transcribe_gemini(audio_path: str) -> list[TranscriptSegment]:
    import base64
    from llm.gemini_client import get_client, get_gemini_model as _model

    audio_bytes = Path(audio_path).read_bytes()
    b64_audio   = base64.b64encode(audio_bytes).decode("utf-8")

    # Detect mime type from extension
    ext = Path(audio_path).suffix.lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    mime_type = mime_map.get(ext, "audio/mpeg")

    from google.genai import types

    client = get_client()
    response = client.models.generate_content(
        model=_model(),
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            types.Part.from_text(

            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )

    return _parse_gemini_transcript(response.text or "")


def _parse_gemini_transcript(text: str) -> list[TranscriptSegment]:
    import re

    segments: list[TranscriptSegment] = []
    pattern = re.compile(r"^\[(\d{1,2}):(\d{2})\]\s*(.+)$")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    prev_start = 0.0

    for line in lines:
        match = pattern.match(line)
        if match:
            m, s, content = match.groups()
            start = int(m) * 60 + int(s)
            if segments:
                # Close previous segment's end time
                segments[-1] = TranscriptSegment(
                    start=segments[-1].start,
                    end=start,
                    text=segments[-1].text,
                )
            segments.append(TranscriptSegment(
                start=float(start),
                end=float(start) + 30.0,
                text=content.strip(),
            ))
            prev_start = float(start)
        elif segments and line:
            # Continuation line — append to last segment
            last = segments[-1]
            segments[-1] = TranscriptSegment(
                start=last.start,
                end=last.end,
                text=last.text + " " + line,
            )

    # Fallback: no timestamps found — treat as one big segment
    if not segments and text.strip():
        segments.append(TranscriptSegment(
            start=0.0, end=30.0, text=text.strip()
        ))

    return segments


def _transcribe_whisper(audio_path: str, model_size: str) -> list[TranscriptSegment]:

    from functools import lru_cache
    import whisper

    @lru_cache(maxsize=1)
    def _load(size: str):
        return whisper.load_model(size)

    model  = _load(model_size)
    result = model.transcribe(audio_path, verbose=False, task="transcribe")

    segments: list[TranscriptSegment] = []
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

    if is_cloud():
        return _transcribe_gemini(audio_path)
 
    size = model_size or get_whisper_model()
    return _transcribe_whisper(audio_path, size)