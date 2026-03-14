import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import re
from typing import Callable
from models import TranscriptSegment


def _timecode_to_seconds(tc: str) -> float:
    tc = tc.replace(",", ".")
    parts = tc.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(tc)


def _parse_txt(text: str) -> list:
    segments = []
    ts_pattern = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*")
    current_start = 0.0
    buffer = []

    def _flush(start, lines):
        combined = " ".join(lines).strip()
        if combined:
            segments.append(TranscriptSegment(start=start, end=start + 30.0, text=combined))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = ts_pattern.match(line)
        if match:
            _flush(current_start, buffer)
            buffer = [ts_pattern.sub("", line)]
            h = int(match.group(3) or 0)
            m = int(match.group(1))
            s = int(match.group(2))
            current_start = h * 3600 + m * 60 + s
        else:
            buffer.append(line)
    _flush(current_start, buffer)
    return segments


def _parse_srt(text: str) -> list:
    segments = []
    blocks = re.split(r"\n\s*\n", text.strip())
    TC = r"(\d{2}:\d{2}:\d{2}[,\.]\d+)"
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        match = re.match(rf"{TC}\s*-->\s*{TC}", lines[1])
        if not match:
            continue
        start = _timecode_to_seconds(match.group(1))
        end = _timecode_to_seconds(match.group(2))
        content = " ".join(lines[2:]).strip()
        if content:
            segments.append(TranscriptSegment(start=start, end=end, text=content))
    return segments


def _parse_vtt(text: str) -> list:
    segments = []
    text = re.sub(r"^WEBVTT.*?\n\n", "", text, flags=re.DOTALL)
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith("NOTE"):
            continue
        for i, line in enumerate(lines):
            match = (
                re.match(r"(\d{2}:\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d+)", line) or
                re.match(r"(\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}\.\d+)", line)
            )
            if match:
                start = _timecode_to_seconds(match.group(1))
                end = _timecode_to_seconds(match.group(2))
                content = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
                if content:
                    segments.append(TranscriptSegment(start=start, end=end, text=content))
                break
    return segments


# _PARSERS must be defined AFTER all parser functions above
_PARSERS = {
    ".srt": _parse_srt,
    ".vtt": _parse_vtt,
    ".txt": _parse_txt,
}


def load_file(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    suffix = p.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported format: '{suffix}'. Supported: {', '.join(_PARSERS)}")
    return parser(p.read_text(encoding="utf-8", errors="replace"))