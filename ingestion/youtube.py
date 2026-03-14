import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import shutil
from pathlib import Path
import yt_dlp

from config import get_ffmpeg_path


    

def _sanitize_filename(name: str) -> str:
    """Strip characters unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def _find_ffmpeg_dir() -> str:

    env_val = get_ffmpeg_path()
    if env_val:
        p = Path(env_val)
        if p.is_file():
            return str(p.parent)
        if p.is_dir() and (p / "ffmpeg").exists():
            return str(p)
        # Windows, check for ffmpeg.exe
        if p.is_dir() and (p / "ffmpeg.exe").exists():
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
        "Or set FFMPEG_PATH=/path/to/ffmpeg/dir in your .env file."
    )


def _find_node() -> str | None:

    for name in ("node", "nodejs"):
        found = shutil.which(name)
        if found:
            return found
    for path in ("/usr/bin/node", "/usr/local/bin/node", "/opt/homebrew/bin/node"):
        if Path(path).exists():
            return path
    return None


def _build_extractor_args() -> dict:
    node = _find_node()
    if node:
        return {"youtube": {"js_runtimes": [f"nodejs:{node}"]}}
    return {}



def download_audio(url: str, output_dir: str = "data") -> tuple[str, str]:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ffmpeg_dir      = _find_ffmpeg_dir()
    extractor_args  = _build_extractor_args()

    original_path   = os.environ.get("PATH", "")
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + original_path

    try:
        probe_opts: dict = {"quiet": True, "no_warnings": True}
        if extractor_args:
            probe_opts["extractor_args"] = extractor_args

        with yt_dlp.YoutubeDL(probe_opts) as probe:
            info  = probe.extract_info(url, download=False)
            title = _sanitize_filename(info.get("title", "video"))

        ydl_opts: dict = {
            "format":           "bestaudio/best",
            "outtmpl":          str(Path(output_dir) / f"{title}.%(ext)s"),
            "postprocessors":   [{
                "key":              "FFmpegExtractAudio",
                "preferredcodec":   "mp3",
                "preferredquality": "128",
            }],
            "ffmpeg_location":  ffmpeg_dir,
            "quiet":            True,
            "no_warnings":      True,
        }
        if extractor_args:
            ydl_opts["extractor_args"] = extractor_args

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    finally:
        os.environ["PATH"] = original_path  

    audio_path = Path(output_dir) / f"{title}.mp3"
    if not audio_path.exists():
        raise ValueError(
            f"Download failed — expected output not found: {audio_path}\n"
            "The video may be age-restricted, private, or region-blocked."
        )

    return str(audio_path), title