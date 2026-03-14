import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel



@dataclass
class TranscriptSegment:
    """One timestamped segment from Whisper or a parsed .srt/.vtt file."""
    start: float          # seconds
    end: float            
    text: str

    @property
    def timestamp_str(self) -> str:
        """Human-readable timestamp e.g. [03:45]"""
        minutes = int(self.start // 60)
        seconds = int(self.start % 60)
        return f"[{minutes:02d}:{seconds:02d}]"


@dataclass
class Chunk:
    """A chunk of transcript text ready for embedding."""
    chunk_id: str
    timestamp_str: str    # e.g. "[03:45]"
    start: float          # seconds, for ordering
    text: str



class Note(BaseModel):
    timestamp: Optional[str] = None   # e.g. "[03:45]"
    heading: str
    content: str


class ExtractionResult(BaseModel):
    title: str
    summary: str
    notes: list[Note]
    action_items: list[str]
    key_concepts: list[str]