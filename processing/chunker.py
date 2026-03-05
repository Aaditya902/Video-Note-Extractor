import uuid
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import TranscriptSegment, Chunk


def chunk_segments(
    segments: list[TranscriptSegment],
    window_words: int = 250,
    overlap_words: int = 50,
) -> list[Chunk]:

    if not segments:
        return []

    chunks: list[Chunk] = []
    seg_index = 0
    n = len(segments)

    while seg_index < n:
        # Build a chunk starting from seg_index
        word_count = 0
        chunk_segs: list[TranscriptSegment] = []
        i = seg_index

        while i < n and word_count < window_words:
            chunk_segs.append(segments[i])
            word_count += len(segments[i].text.split())
            i += 1

        if not chunk_segs:
            break

        anchor = chunk_segs[0]
        text = " ".join(s.text for s in chunk_segs)

        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            timestamp_str=anchor.timestamp_str,
            start=anchor.start,
            text=text,
        ))

        # Advance by (window - overlap) words — find which segment to jump to
        words_to_skip = window_words - overlap_words
        skipped = 0
        while seg_index < n and skipped < words_to_skip:
            skipped += len(segments[seg_index].text.split())
            seg_index += 1

    return chunks