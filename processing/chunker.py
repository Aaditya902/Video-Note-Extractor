import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uuid
from models import TranscriptSegment, Chunk

    
def _word_count(segment: TranscriptSegment) -> int:
    return len(segment.text.split())


def chunk_segments(
    segments: list[TranscriptSegment],
    window_words: int = 250,
    overlap_words: int = 50,
) -> list[Chunk]:

    if not segments:
        return []

    chunks: list[Chunk] = []
    n         = len(segments)
    seg_index = 0

    while seg_index < n:
        word_count  = 0
        chunk_segs: list[TranscriptSegment] = []
        i = seg_index

        while i < n and word_count < window_words:
            chunk_segs.append(segments[i])
            word_count += _word_count(segments[i])
            i += 1

        if not chunk_segs:
            break

        anchor = chunk_segs[0]
        chunks.append(Chunk(
            chunk_id      = str(uuid.uuid4()),
            timestamp_str = anchor.timestamp_str,
            start         = anchor.start,
            text          = " ".join(s.text for s in chunk_segs),
        ))

        words_to_skip = window_words - overlap_words
        skipped       = 0
        while seg_index < n and skipped < words_to_skip:
            skipped   += _word_count(segments[seg_index])
            seg_index += 1

    return chunks