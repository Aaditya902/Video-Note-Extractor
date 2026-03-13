from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from models import ExtractionResult, TranscriptSegment
from processing.vector_store import VectorStore


class InputType(str, Enum):
    VIDEO = "video"
    YOUTUBE = "youtube"
    FILE = "file"


@dataclass
class PipelineResult:
    result: ExtractionResult
    store: VectorStore
    title: str

@dataclass
class PipelineProgress:
    completed: list[str] = field(default_factory=list)
    active:    str  | None = None
    error:     str  | None = None


ProgressCallback = Callable[[PipelineProgress], None]

STEPS = [
    ("ingest",  "Ingestion input"),
    ("transcribe", "Transcribing with Whisper"),
    ("chunk", "chunking transcript"),
    ("embed", "Embedding chunks"),
    ("llm", "Extracting notes with Gemini"),
]

def _ingest(
    input_type: InputType,
    input_data,
    tmp_dir: str,
    on_progress: ProgressCallback,
    completed: list[str],
) -> tuple[list[TranscriptSegment], str]:

    if input_type == InputType.VIDEO:
        from ingestion.local_video import extract_audio
        from transcription.whisper_engine import transcribe

        audio_path, meta = extract_audio(input_data, output_dir = tmp_dir)
        title = meta.title
        completed.append("ingest")
        on_progress(PipelineProgress(completed=list(completed), active = "transcribe"))

        segments = transcribe(audio_path)
        completed.append("transcribe")

    elif input_type == InputType.YOUTUBE:
        from ingestion.local_video import extract_audio
        from transcription.whisper_engine import transcribe

        audio_path, title = download_audio(input_data, output_dir = tmp_dir)
        completed.append("ingest")
        on_progress(PipelineProgress(completed=list(completed), active = "transcribe"))

        segments = transcribe(audio_path)
        completed.append("transcribe")

    elif input_type == InputType.FILE:
        from ingestion.file_loader import load_file

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

def run(
    input_type: InputType,
    input_data,
    whisper_model: str = "base",
    on_progress: ProgressCallback | None = None,
) -> PipelineResult:

    if on_progress is None:
        on_progress = lambda _: None   # no-op default
 
    completed: list[str] = []
    tmp_dir = tempfile.mkdtemp()
 

    on_progress(PipelineProgress(completed=[], active="ingest"))
    segments, title = _ingest(input_type, input_data, tmp_dir, on_progress, completed)
    on_progress(PipelineProgress(completed=list(completed), active="chunk"))
 
    if not segments:
        raise ValueError("No transcript segments found in this input.")
 
    
    from processing.chunker import chunk_segments
    chunks = chunk_segments(segments, window_words=250, overlap_words=50)
    completed.append("chunk")
    on_progress(PipelineProgress(completed=list(completed), active="embed"))
 
    
    store = VectorStore()
    store.add_chunks(chunks)
    completed.append("embed")
    on_progress(PipelineProgress(completed=list(completed), active="llm"))
 

    from llm.gemini_extractor import extract_notes
    extraction = extract_notes(store, video_title=title)
    completed.append("llm")
    on_progress(PipelineProgress(completed=list(completed)))
 
    return PipelineResult(result=extraction, store=store, title=title)



