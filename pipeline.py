import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from models import ExtractionResult, TranscriptSegment
from processing.vector_store import VectorStore
from processing.chunker import chunk_segments
from llm.gemini_extractor import extract_notes
from ingestion.local_video import extract_audio
from ingestion.youtube import download_audio
from ingestion.file_loader import load_file
from transcription.whisper_engine import transcribe


class InputType(str, Enum):
    VIDEO   = "video"
    YOUTUBE = "youtube"
    FILE    = "file"


@dataclass
class PipelineResult:
    result: ExtractionResult
    store:  VectorStore
    title:  str


@dataclass
class PipelineProgress:
    completed: list = field(default_factory=list)
    active:    Optional[str] = None
    error:     Optional[str] = None


ProgressCallback = Callable[[PipelineProgress], None]

STEPS = [
    ("ingest",     "Ingesting input"),
    ("transcribe", "Transcribing with Whisper"),
    ("chunk",      "Chunking transcript"),
    ("embed",      "Embedding chunks"),
    ("llm",        "Extracting notes with Gemini"),
]


def _ingest(input_type, input_data, tmp_dir, on_progress, completed):
    if input_type == InputType.VIDEO:
        audio_path, meta = extract_audio(input_data, output_dir=tmp_dir)
        title = meta.title
        completed.append("ingest")
        on_progress(PipelineProgress(completed=list(completed), active="transcribe"))
        segments = transcribe(audio_path)
        completed.append("transcribe")

    elif input_type == InputType.YOUTUBE:
        audio_path, title = download_audio(input_data, output_dir=tmp_dir)
        completed.append("ingest")
        on_progress(PipelineProgress(completed=list(completed), active="transcribe"))
        segments = transcribe(audio_path)
        completed.append("transcribe")

    elif input_type == InputType.FILE:
        fname, fbytes = input_data
        tmp_path = Path(tmp_dir) / fname
        tmp_path.write_bytes(fbytes)
        segments = load_file(str(tmp_path))
        title = Path(fname).stem
        completed.append("ingest")
        completed.append("transcribe")

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    return segments, title


def run(input_type, input_data, whisper_model="base", on_progress=None):
    if on_progress is None:
        on_progress = lambda _: None

    completed = []
    tmp_dir = tempfile.mkdtemp()

    on_progress(PipelineProgress(completed=[], active="ingest"))
    segments, title = _ingest(input_type, input_data, tmp_dir, on_progress, completed)
    on_progress(PipelineProgress(completed=list(completed), active="chunk"))

    if not segments:
        raise ValueError("No transcript segments found in this input.")

    chunks = chunk_segments(segments, window_words=250, overlap_words=50)
    completed.append("chunk")
    on_progress(PipelineProgress(completed=list(completed), active="embed"))

    store = VectorStore()
    store.add_chunks(chunks)
    completed.append("embed")
    on_progress(PipelineProgress(completed=list(completed), active="llm"))

    extraction = extract_notes(store, video_title=title)
    completed.append("llm")
    on_progress(PipelineProgress(completed=list(completed)))

    return PipelineResult(result=extraction, store=store, title=title)