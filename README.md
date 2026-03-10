# 🎬 Video Note Extractor

> Convert YouTube videos, lectures, and meetings into **organized notes**, **timestamps**, and **action items** using local Whisper + RAG + Claude.

---

## Architecture

```
Input (YouTube URL or .txt/.srt/.vtt file)
    │
    ▼
[Ingestion]        yt-dlp downloads audio  OR  file_loader parses transcript
    │
    ▼
[Transcription]    Whisper (local) → List[TranscriptSegment{start, end, text}]
    │              (skipped if transcript file provided)
    ▼
[Chunker]          Sliding window (250 words, 50 overlap) → List[Chunk]
    │
    ▼
[Embedder]         sentence-transformers all-MiniLM-L6-v2 → embeddings
    │
    ▼
[ChromaDB]         In-memory vector store, cosine similarity
    │
    ▼  (multi-query RAG retrieval)
[Claude]           claude-sonnet → structured JSON notes
    │
    ▼
[Output]           notes.md  +  notes.json  →  saved to ./data/
```

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/Aaditya902/Video-Note-Extractor.git  && cd video-note-extractor

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY

# 5. Run
python main.py --file lecture.srt
python main.py --url "https://youtube.com/watch?v=..."
```

## Usage

```bash
# From a transcript file
python main.py --file path/to/lecture.srt

# From YouTube
python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Use a more accurate Whisper model (slower)
python main.py --url "..." --whisper-model small

# Custom output directory
python main.py --file transcript.txt --output ./my_notes
```

## Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Optional `[MM:SS]` timestamps |
| SubRip subtitles | `.srt` | Standard subtitle format |
| WebVTT captions | `.vtt` | YouTube auto-caption export |
| YouTube URL | `--url` | Requires `ffmpeg` for audio extraction |

## Project Structure

```
video-note-extractor/
├── main.py                   # CLI entry point
├── models.py                 # Shared data models (Pydantic + dataclasses)
├── requirements.txt
├── .env.example
├── ingestion/
│   ├── youtube.py            # yt-dlp audio download
│   └── file_loader.py        # .txt / .srt / .vtt parser
├── transcription/
│   └── whisper_engine.py     # Whisper speech-to-text
├── processing/
│   ├── chunker.py            # Sliding window chunker
│   ├── embedder.py           # sentence-transformers
│   └── vector_store.py       # ChromaDB wrapper
├── llm/
│   └── claude_extractor.py   # Multi-query RAG + Claude
└── output/
    └── formatter.py          # Markdown + JSON writer
```

## Whisper Model Guide

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | 39M  | Fast  | Basic |
| base  | 74M  | Good  | Good ← **default** |
| small | 244M | Slower| Better |
| medium| 769M | Slow  | Great |
| large | 1.5G | Slow  | Best |


- [ ] Streamlit web UI
- [ ] Batch processing (multiple videos)
- [ ] Export to Notion / Obsidian
- [ ] Support for local video files (.mp4)
- [ ] Persistent ChromaDB for cross-session search