"""
ingestion/youtube.py — Fetch YouTube video transcript via Gemini native URL support.

Key insight: Gemini supports YouTube URLs natively via file_data.file_uri.
Google fetches the video on their own infrastructure — completely bypassing
the cloud IP blocking issue (HTTP 403) that affects yt-dlp and
youtube-transcript-api on shared cloud servers.

No yt-dlp, no FFmpeg, no audio download needed for YouTube URLs.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import re

from models import TranscriptSegment


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any valid YouTube URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        f"Could not extract video ID from URL: {url}\n"
        "Supported formats: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID"
    )


def _parse_transcript(text: str) -> list[TranscriptSegment]:
    """Parse [MM:SS] timestamped lines into TranscriptSegments."""
    segments: list[TranscriptSegment] = []
    pattern = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.+)$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            a, b, c, content = match.groups()
            start = int(a) * 3600 + int(b) * 60 + int(c or 0) if c else int(a) * 60 + int(b)
            if segments:
                prev = segments[-1]
                segments[-1] = TranscriptSegment(start=prev.start, end=float(start), text=prev.text)
            segments.append(TranscriptSegment(
                start=float(start), end=float(start) + 30.0, text=content.strip()
            ))
        elif segments:
            last = segments[-1]
            segments[-1] = TranscriptSegment(
                start=last.start, end=last.end, text=last.text + " " + line
            )

    if not segments and text.strip():
        segments.append(TranscriptSegment(start=0.0, end=30.0, text=text.strip()))

    return segments


def fetch_transcript(url: str) -> tuple[list[TranscriptSegment], str]:
    """
    Fetch YouTube video transcript using Gemini's native YouTube URL support.

    Gemini accepts YouTube URLs directly via file_data.file_uri — Google
    fetches the video on their own servers, completely bypassing cloud IP
    blocking. No yt-dlp, no FFmpeg, no audio download required.

    Args:
        url: Any valid YouTube URL

    Returns:
        (segments, video_title)

    Raises:
        ValueError:       invalid URL or Gemini cannot access the video
        EnvironmentError: GEMINI_API_KEY not set
    """
    from llm.gemini_client import get_client, get_gemini_model
    from google.genai import types

    video_id = _extract_video_id(url)
    clean_url = f"https://www.youtube.com/watch?v={video_id}"

    prompt = (
        "You are a transcription and analysis assistant. "
        "Watch this YouTube video and do two things:\n\n"
        "1. TITLE: Output the video title on the first line as: TITLE: <title here>\n\n"
        "2. TRANSCRIPT: Transcribe all spoken content completely.\n"
        "   Format every line as: [MM:SS] transcript text here\n"
        "   Example:\n"
        "   [00:00] Welcome to this tutorial.\n"
        "   [00:30] Today we will cover three main topics.\n"
        "   [01:45] Let's start with the first concept.\n\n"
        "Rules:\n"
        "- Every line MUST start with [MM:SS] timestamp\n"
        "- New timestamp every 20-30 seconds\n"
        "- Include ALL spoken content, nothing omitted\n"
        "- Output ONLY the title line and transcript lines, nothing else"
    )

    client   = get_client()
    model    = get_gemini_model()

    video_part = types.Part.from_uri(
        file_uri  = clean_url,
        mime_type = "video/mp4",
    )

    response = client.models.generate_content(
        model    = model,
        contents = [
            video_part,
            types.Part.from_text(text=prompt),
        ],
    )

    raw = response.text or ""

    # Extract title from first line if present
    title = "YouTube Video"
    lines = raw.strip().splitlines()
    transcript_lines = []

    for line in lines:
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()
        else:
            transcript_lines.append(line)

    transcript_text = "\n".join(transcript_lines)
    segments = _parse_transcript(transcript_text)

    if not segments:
        raise ValueError(
            "Gemini could not transcribe this video.\n"
            "The video may be private, age-restricted, or have no speech."
        )

    return segments, title