import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import shutil

import yt_dlp

from config import get_ffmpeg_path, is_cloud


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def _ensure_ffmpeg() -> None:

    import subprocess
    # Already installed?
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(p).exists():
            return
    if shutil.which("ffmpeg"):
        return
    # Try to install via apt (Streamlit Cloud is Ubuntu)
    try:
        subprocess.run(
            ["apt-get", "install", "-y", "ffmpeg"],
            check=True, capture_output=True
        )
    except Exception:
        try:
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "ffmpeg"],
                check=True, capture_output=True
            )
        except Exception:
            pass  # Will raise a clear error in _find_ffmpeg_dir


def _find_ffmpeg_dir() -> str:
    env_val = get_ffmpeg_path()
    if env_val:
        p = Path(env_val)
        if p.is_file():
            return str(p.parent)
        if p.is_dir() and (p / "ffmpeg").exists():
            return str(p)


    for loc in [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
        "/bin/ffmpeg",
        "/snap/bin/ffmpeg",
    ]:
        if Path(loc).exists():
            return str(Path(loc).parent)

    # 3. PATH lookup
    binary = shutil.which("ffmpeg")
    if binary:
        return str(Path(binary).parent)

    # 4. Last resort: try installing at runtime (Streamlit Cloud fallback)
    _ensure_ffmpeg()

    # Check again after install attempt
    for loc in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(loc).exists():
            return str(Path(loc).parent)
    binary = shutil.which("ffmpeg")
    if binary:
        return str(Path(binary).parent)

    raise EnvironmentError(
        "ffmpeg not found and could not be installed automatically.\n"
        "  Streamlit Cloud: make sure packages.txt contains 'ffmpeg' "
        "in the root of your repository and trigger a full reboot "
        "(Settings → Reboot app).\n"
        "  Local: brew install ffmpeg  /  sudo apt install ffmpeg"
    )


def _find_node() -> str | None:
    for name in ("node", "nodejs"):
        found = shutil.which(name)
        if found:
            return found
    for p in ("/usr/bin/node", "/usr/local/bin/node", "/opt/homebrew/bin/node"):
        if Path(p).exists():
            return p
    return None


def _ydl_common_opts(ffmpeg_dir: str, extractor_args: dict) -> dict:

    opts = {
        "quiet":        True,
        "no_warnings":  True,
        # Bypass options — help on cloud IPs that YouTube rate-limits
        "nocheckcertificate": True,
        "geo_bypass":         True,
        # Use Android client — less likely to be blocked than web client
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        # Retry on failure
        "retries":            5,
        "fragment_retries":   5,
    }
    # Override extractor_args if node is available (adds JS runtime)
    if extractor_args:
        opts["extractor_args"] = {
            **opts["extractor_args"],
            **extractor_args,
        }
    return opts


def download_audio(url: str, output_dir: str = "data") -> tuple[str, str]:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = _find_ffmpeg_dir()

    node = _find_node()
    node_args = {"youtube": {"js_runtimes": [f"nodejs:{node}"]}} if node else {}

    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + original_path

    try:
        common = _ydl_common_opts(ffmpeg_dir, node_args)

        try:
            with yt_dlp.YoutubeDL(common) as probe:
                info  = probe.extract_info(url, download=False)
                title = _sanitize(info.get("title", "video"))
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["sign in", "bot", "403", "429", "blocked", "unavailable"]):
                raise ValueError(
                    "YouTube blocked this download.\n\n"
                    "This is a known limitation when running on cloud platforms "
                    "(Hugging Face, Streamlit Cloud, etc.) — YouTube rate-limits "
                    "requests from shared cloud IPs.\n\n"
                    "✅ Workarounds:\n"
                    "  1. Download the video locally and upload it as a video file\n"
                    "  2. Download the transcript (.srt) from YouTube and upload it\n"
                    "  3. Run this app locally where your IP is not blocked"
                ) from e
            raise

        ydl_opts = {
            **common,
            "format":          "bestaudio/best",
            "outtmpl":         str(Path(output_dir) / f"{title}.%(ext)s"),
            "postprocessors":  [{
                "key":              "FFmpegExtractAudio",
                "preferredcodec":   "mp3",
                "preferredquality": "128",
            }],
            "ffmpeg_location": ffmpeg_dir,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    finally:
        os.environ["PATH"] = original_path

    audio_path = Path(output_dir) / f"{title}.mp3"
    if not audio_path.exists():
        raise ValueError(
            f"Download completed but output file not found: {audio_path}\n"
            "The video may be age-restricted, private, or region-blocked."
        )

    return str(audio_path), title