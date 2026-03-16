import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from config import get_ffmpeg_path

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}


@dataclass
class VideoMetadata:
    title: str
    duration_sec: float
    has_audio: bool
    video_codec: str
    audio_codec: str
    format_name: str


def _resolve_binary(name: str) -> str:
    # 1. Explicit env override
    env_dir = get_ffmpeg_path()
    if env_dir:
        candidate = Path(env_dir) / name
        if candidate.exists():
            return str(candidate)

    # 2. Check known locations first (more reliable inside Docker/non-root)
    for d in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/opt/local/bin",
              r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin"]:
        p = Path(d) / name
        if p.exists():
            return str(p)

    # 3. PATH lookup fallback
    found = shutil.which(name)
    if found:
        return found

    raise EnvironmentError(
        f"'{name}' not found. Install FFmpeg:\n"
        "  macOS:   brew install ffmpeg\n"
        "  Ubuntu:  sudo apt install ffmpeg\n"
        "  Docker:  add RUN apt-get install -y ffmpeg to your Dockerfile"
    )


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def probe_video(path: str) -> VideoMetadata:
    cmd = [_resolve_binary("ffprobe"), "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_codec = next((s["codec_name"] for s in streams if s.get("codec_type") == "video"), "unknown")
    audio_codec = next((s["codec_name"] for s in streams if s.get("codec_type") == "audio"), None)
    return VideoMetadata(
        title=_sanitize(Path(path).stem),
        duration_sec=float(fmt.get("duration", 0.0)),
        has_audio=audio_codec is not None,
        video_codec=video_codec,
        audio_codec=audio_codec or "none",
        format_name=fmt.get("format_name", "unknown"),
    )


def extract_audio(video_path: str, output_dir: str = "data") -> tuple[str, VideoMetadata]:
    p = Path(video_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported type: '{p.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    meta = probe_video(str(p))
    if not meta.has_audio:
        raise ValueError(f"'{p.name}' has no audio stream.")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_path = Path(output_dir) / f"{meta.title}_audio.wav"
    if audio_path.exists():
        return str(audio_path), meta
    cmd = [_resolve_binary("ffmpeg"), "-i", str(p), "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-y", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-1000:]}")
    if not audio_path.exists():
        raise RuntimeError(f"FFmpeg exited cleanly but output missing: {audio_path}")
    return str(audio_path), meta