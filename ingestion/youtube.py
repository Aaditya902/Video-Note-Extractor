"""
ingestion/youtube.py - Download audio from a YouTube URL using yt-dlp.

Fixes applied:
  1. JS runtime: explicitly passes Node.js path to yt-dlp so YouTube
     extraction works without the deprecation warning.
  2. FFmpeg: injects the resolved ffmpeg directory into both PATH (for
     yt-dlp's availability check) and ffmpeg_location (for postprocessors).
"""

import os
import re
import shutil
from pathlib import Path

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Strip characters that are unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def _find_ffmpeg() -> str:
    """
    Return the DIRECTORY containing the ffmpeg binary.
    (yt-dlp's ffmpeg_location takes a dir, not the binary path itself.)

    Search order:
      1. FFMPEG_PATH env var
      2. shutil.which() - PATH lookup
      3. Hard-coded fallbacks
    """
    env_val = os.environ.get("FFMPEG_PATH", "")
    if env_val:
        p = Path(env_val)
        if p.is_file():
            return str(p.parent)
        if p.is_dir() and (p / "ffmpeg").exists():
            return str(p)

    binary = shutil.which("ffmpeg")
    if binary:
        return str(Path(binary).parent)

    for d in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/opt/local/bin"]:
        if Path(d, "ffmpeg").exists():
            return d

    raise EnvironmentError(
        "ffmpeg not found.\n"
        "  macOS:   brew install ffmpeg\n"
        "  Ubuntu:  sudo apt install ffmpeg\n"
        "  Windows: https://www.gyan.dev/ffmpeg/builds/\n"
        "Or add FFMPEG_PATH=/usr/bin to your .env file."
    )


def _find_node() -> str | None:
    """
    Return the full path to the Node.js binary, or None if not installed.
    yt-dlp needs a JS runtime (node/deno) for modern YouTube extraction.
    """
    for name in ("node", "nodejs"):
        found = shutil.which(name)
        if found:
            return found
    for path in ("/usr/bin/node", "/usr/local/bin/node", "/opt/homebrew/bin/node"):
        if Path(path).exists():
            return path
    return None


def download_audio(url: str, output_dir: str = "data") -> tuple[str, str]:
    """
    Download audio from a YouTube URL and convert to mp3 via FFmpeg.

    Args:
        url:        YouTube video URL
        output_dir: Directory to save the audio file

    Returns:
        (audio_file_path, video_title)

    Raises:
        EnvironmentError: if ffmpeg is not found
        ValueError:       if download output file is missing after run
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Resolve FFmpeg ────────────────────────────────────────────────────────
    # Belt-and-suspenders: inject into PATH *and* pass via ffmpeg_location.
    # yt-dlp does an availability check against PATH separately from
    # the ffmpeg_location postprocessor option — both need to be set.
    ffmpeg_dir = _find_ffmpeg()
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + original_path

    # ── Resolve JS runtime ────────────────────────────────────────────────────
    # yt-dlp defaults to deno-only. Node.js is available, so pass it explicitly.
    # Format: "nodejs:/full/path/to/node"
    node_bin = _find_node()
    js_runtimes = [f"nodejs:{node_bin}"] if node_bin else []

    # ── Build shared extractor args ───────────────────────────────────────────
    extractor_args: dict = {}
    if js_runtimes:
        extractor_args = {"youtube": {"js_runtimes": js_runtimes}}

    try:
        # ── Metadata probe (no download) ──────────────────────────────────────
        probe_opts: dict = {
            "quiet": True,
            "no_warnings": True,
        }
        if extractor_args:
            probe_opts["extractor_args"] = extractor_args

        with yt_dlp.YoutubeDL(probe_opts) as probe:
            info  = probe.extract_info(url, download=False)
            title = sanitize_filename(info.get("title", "video"))

        output_template = str(Path(output_dir) / f"{title}.%(ext)s")

        # ── Download + postprocess ────────────────────────────────────────────
        ydl_opts: dict = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",  # 128kbps — sufficient for speech
                }
            ],
            "ffmpeg_location": ffmpeg_dir,  # directory, not binary path
            "quiet": True,
            "no_warnings": True,
        }
        if extractor_args:
            ydl_opts["extractor_args"] = extractor_args

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    finally:
        # Always restore original PATH regardless of success/failure
        os.environ["PATH"] = original_path

    audio_path = str(Path(output_dir) / f"{title}.mp3")

    if not Path(audio_path).exists():
        raise ValueError(
            f"Download failed — expected output not found: {audio_path}\n"
            "The video may be age-restricted, private, or region-blocked."
        )

    return audio_path, title