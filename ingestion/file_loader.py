import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import TranscriptSegment


def load_file(path: str) -> list[TranscriptSegment]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".srt":
        return _parse_srt(text)
    elif suffix == ".vtt":
        return _parse_vtt(text)
    else:
        return _parse_txt(text)


def _parse_txt(text: str) -> list[TranscriptSegment]:
    """
    Parse plain text. Recognises optional [MM:SS] or [HH:MM:SS] prefixes.
    Lines without timestamps are grouped under the last seen timestamp.
    """
    segments: list[TranscriptSegment] = []
    timestamp_pattern = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*")

    current_start = 0.0
    buffer: list[str] = []

    def flush(start: float, lines: list[str]) -> None:
        combined = " ".join(lines).strip()
        if combined:
            segments.append(TranscriptSegment(start=start, end=start + 30.0, text=combined))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = timestamp_pattern.match(line)
        if match:
            flush(current_start, buffer)
            buffer = [timestamp_pattern.sub("", line)]
            h = int(match.group(3) or 0)
            m = int(match.group(1))
            s = int(match.group(2))
            current_start = h * 3600 + m * 60 + s
        else:
            buffer.append(line)

    flush(current_start, buffer)
    return segments


def _timecode_to_seconds(tc: str) -> float:
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to float seconds."""
    tc = tc.replace(",", ".")
    parts = tc.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(tc)


def _parse_srt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # lines[0] = block index (skip)
        # lines[1] = "00:00:01,000 --> 00:00:04,000"
        # lines[2:] = subtitle text
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)",
            lines[1]
        )
        if not time_match:
            continue
        start = _timecode_to_seconds(time_match.group(1))
        end = _timecode_to_seconds(time_match.group(2))
        content = " ".join(lines[2:]).strip()
        if content:
            segments.append(TranscriptSegment(start=start, end=end, text=content))

    return segments


def _parse_vtt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    # Strip WEBVTT header
    text = re.sub(r"^WEBVTT.*?\n\n", "", text, flags=re.DOTALL)
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        # Skip NOTE or empty blocks
        if not lines or lines[0].startswith("NOTE"):
            continue
        # Find the timecode line
        for i, line in enumerate(lines):
            time_match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d+)",
                line
            )
            if not time_match:
                time_match = re.match(
                    r"(\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}\.\d+)", line
                )
            if time_match:
                start = _timecode_to_seconds(time_match.group(1))
                end = _timecode_to_seconds(time_match.group(2))
                content = " ".join(lines[i + 1:]).strip()
                content = re.sub(r"<[^>]+>", "", content).strip()
                if content:
                    segments.append(TranscriptSegment(start=start, end=end, text=content))
                break

    return segments