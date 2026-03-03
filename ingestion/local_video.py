import json
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VideoMetadata:
    title: str          # filename stem (no extension)
    duration_sec: float # total duration in seconds
    has_audio: bool
    video_codec: str
    audio_codec: str
    format_name: str


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}

def _resolve_binary(name: str) -> str:

    env_dir = os.environ.get("FFMPEG_PATH", "")
    if env_dir:
        candidate = Path(env_dir) / name
        if candidate.exists():
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    fallbacks = [
        "/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/opt/local/bin",
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
    ]
    for d in fallbacks:
        p = Path(d) / name
        if p.exists():
            return str(p)

    raise EnvironmentError(
        f"'{name}' not found. Install FFmpeg:\n"
        "  macOS:   brew install ffmpeg\n"
        "  Ubuntu:  sudo apt install ffmpeg\n"
        "  Windows: https://www.gyan.dev/ffmpeg/builds/\n"
        "Or set FFMPEG_PATH=/usr/bin in your .env file."
    )


def _require_ffmpeg() -> None:
    _resolve_binary("ffmpeg")
    _resolve_binary("ffprobe")


def _sanitize(name: str) -> str:
    """Strip characters unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def probe_video(path: str) -> VideoMetadata:
eError if the file has no audio stream.
    """
    cmd = [
        _resolve_binary("ffprobe"),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(
            f"ffprobe failed on '{path}'.\n"
            f"stderr: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_codec = next(
        (s["codec_name"] for s in streams if s.get("codec_type") == "video"), "unknown"
    )
    audio_codec = next(
        (s["codec_name"] for s in streams if s.get("codec_type") == "audio"), None
    )
    has_audio = audio_codec is not None
    duration = float(fmt.get("duration", 0.0))
    title = _sanitize(Path(path).stem)

    return VideoMetadata(
        title=title,
        duration_sec=duration,
        has_audio=has_audio,
        video_codec=video_codec,
        audio_codec=audio_codec or "none",
        format_name=fmt.get("format_name", "unknown"),
    )


def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"



def extract_audio(video_path: str, output_dir: str = "data") -> tuple[str, VideoMetadata]:
    """
    Extract audio from a local video file and save as 16kHz mono WAV.

    Args:
        video_path: Absolute or relative path to the video file
        output_dir: Directory to write the extracted audio file

    Returns:
        (audio_file_path, VideoMetadata)

    Raises:
        FileNotFoundError: if the video file doesn't exist
        ValueError:        if the file has no audio stream or ffprobe fails
        EnvironmentError:  if ffmpeg/ffprobe are not installed
        RuntimeError:      if ffmpeg extraction fails
    """
    _require_ffmpeg()

    p = Path(video_path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{p.suffix}'\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    meta = probe_video(str(p))

    if not meta.has_audio:
        raise ValueError(
            f"'{p.name}' has no audio stream. Cannot transcribe a silent video."
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_path = Path(output_dir) / f"{meta.title}_audio.wav"

    # Skip re-extraction if output already exists (idempotent for reruns)
    if audio_path.exists():
        return str(audio_path), meta

    # Flags explained:
    #   -vn          : drop video stream
    #   -acodec pcm_s16le : uncompressed 16-bit PCM (what Whisper wants)
    #   -ar 16000    : resample to 16kHz (Whisper's native sample rate)
    #   -ac 1        : downmix to mono (halves file size, sufficient for speech)
    #   -y           : overwrite without prompt
    cmd = [
        _resolve_binary("ffmpeg"),
        "-i", str(p),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to extract audio from '{p.name}'.\n"
            f"stderr: {result.stderr[-1000:]}"   # last 1000 chars of ffmpeg output
        )

    if not audio_path.exists():
        raise RuntimeError(
            f"FFmpeg exited cleanly but output file missing: {audio_path}"
        )

    return str(audio_path), meta