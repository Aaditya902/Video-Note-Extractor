# 🎬 NoteExtract.ai:  AI-Powered Video Note Extractor

> Convert videos, YouTube links, and transcript files into structured, timestamped notes using a fully local RAG pipeline - with Google Gemini as the only external dependency.


---

## ✨ What It Does

Feed it a video file, a YouTube URL, or a transcript - and get back:

- 📝 **Timestamped notes** with headings and key insights
- 📌 **Executive summary** of the full content
- ✅ **Action items** extracted automatically
- 🏷️ **Key concepts / tags** for quick scanning
- 💬 **Q&A chat** grounded strictly in the extracted notes
- 📥 **Export** as Markdown or JSON - one click, no sign-up

---


### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| FFmpeg | Video/audio processing | See below |
| Gemini API Key | LLM extraction | [aistudio.google.com](https://aistudio.google.com) - free |

**Install FFmpeg:**

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://www.gyan.dev/ffmpeg/builds/
# and add the bin/ folder to your system PATH
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aaditya902/Video-Note-Extractor
cd video-note-extractor

# 2. Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate        # macOS / Linux
myenv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set: GEMINI_API_KEY=your_key_here
```

### Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🗂️ Input Modes

| Mode | Formats | How |
|---|---|---|
| **Local Video** | `.mp4` `.mkv` `.mov` `.avi` `.webm` `.m4v` `.flv` | Drag & drop upload |
| **YouTube URL** | Any public YouTube video | Paste URL |
| **Transcript File** | `.srt` `.vtt` `.txt` | Drag & drop upload |

---

## 🏗️ Architecture

```
Input (video / YouTube / transcript)
         │
         ▼
   ┌─────────────┐
   │  Ingestion  │  FFmpeg · yt-dlp · file parser
   └──────┬──────┘
          │
          ▼
   ┌──────────────────┐
   │  Transcription   │  OpenAI Whisper  (fully local, offline)
   └──────┬───────────┘
          │
          ▼
   ┌────────────┐
   │  Chunking  │  Sliding window - 250 words, 50-word overlap
   └──────┬─────┘
          │
          ▼
   ┌───────────────┐
   │  Embedding    │  all-MiniLM-L6-v2  (fully local, offline)
   └──────┬────────┘
          │
          ▼
   ┌────────────────────┐
   │  ChromaDB          │  In-memory vector store
   │  (Vector Store)    │
   └──────┬─────────────┘
          │
          ├──── Multi-query RAG retrieval ──► Gemini ──► Structured Notes
          │          (5 targeted queries,
          │           dedup + chronological sort)
          │
          └──── Q&A retrieval ──► Gemini ──► Grounded answers
```

---

## 📁 Project Structure

```
video-note-extractor/
│
├── app.py                        # Streamlit UI - pure presentation layer
├── pipeline.py                   # Pipeline orchestration (ingest → extract)
├── config.py                     # All env config - single source of truth
├── models.py                     # Shared Pydantic + dataclass types
├── pyproject.toml                # Package config - clean absolute imports
│
├── ingestion/
│   ├── local_video.py            # FFmpeg audio extraction
│   ├── youtube.py                # yt-dlp YouTube download
│   └── file_loader.py            # .txt / .srt / .vtt parser
│
├── transcription/
│   └── whisper_engine.py         # Local Whisper STT
│
├── processing/
│   ├── chunker.py                # Sliding-window transcript chunker
│   ├── embedder.py               # SentenceTransformers embeddings
│   └── vector_store.py           # ChromaDB in-memory vector store
│
├── llm/
│   ├── gemini_client.py          # Shared Gemini client factory (DRY)
│   ├── gemini_extractor.py       # RAG + Gemini note extraction
│   └── qa_engine.py              # Grounded Q&A engine
│
├── requirements.txt
├── .env.example
└── README.md
```



| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | - | Free key from [aistudio.google.com](https://aistudio.google.com) |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model to use |
| `WHISPER_MODEL` | No | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `FFMPEG_PATH` | No | auto-detected | Path to FFmpeg `bin/` directory |

---


## 💸 Free Tier Limits

This project was built to run entirely for free:

| Service | Cost | Limit |
|---|---|---|
| **Gemini API** | Free | 1,500 req/day · 15 req/min |
| **Whisper** | Free | Unlimited - runs locally |
| **Embeddings** | Free | Unlimited - runs locally |
| **ChromaDB** | Free | Unlimited - in-memory |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Pipeline orchestration | Python (custom) |
| Speech-to-text | OpenAI Whisper (local) |
| Embeddings | SentenceTransformers - `all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (in-memory) |
| LLM | Google Gemini 2.0 Flash (API) |
| Video download | yt-dlp |
| Audio extraction | FFmpeg |
| Data validation | Pydantic v2 |

