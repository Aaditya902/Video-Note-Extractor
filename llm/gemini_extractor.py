import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

from models import ExtractionResult, Note
from processing.vector_store import VectorStore
from llm.gemini_client import generate



RETRIEVAL_QUERIES = [
    "key ideas concepts explanations definitions",
    "important steps process how to instructions",
    "action items tasks recommendations next steps",
    "examples case studies demonstrations",
    "conclusions summary takeaways results",
]

_EXTRACTION_PROMPT = """You are an expert note-taking assistant. You receive segments of a video transcript and must extract structured, high-quality notes.
Be concise but complete. Every note should stand alone as a useful piece of information.
Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation. Do not truncate.

Extract structured notes from these transcript segments:

{context}

Respond with this exact JSON schema:
{{
  "title": "concise title capturing the video's topic",
  "summary": "2-3 sentence executive summary of the entire content",
  "notes": [
    {{
      "timestamp": "[MM:SS] or null if unknown",
      "heading": "short heading (5 words max)",
      "content": "the key insight, concept, or information (1-3 sentences)"
    }}
  ],
  "action_items": [
    "specific, actionable task a viewer should do after watching"
  ],
  "key_concepts": ["term or concept mentioned"]
}}

Rules:
- Extract 6-14 notes covering the full video chronologically
- Notes should be substantive — not just restating the timestamp
- Generate 3-7 clear, specific action items
- List 5-10 key concepts/terms
- Timestamps should match the [MM:SS] markers in the context
- You MUST produce complete, valid JSON — never stop mid-string"""

def _retrieve_chunks(store: VectorStore) -> list[dict]:
    seen: set[str] = set()
    chunks: list[dict] = []

    for query in RETRIEVAL_QUERIES:
        for chunk in store.query(query, top_k = 6):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                chunks.append(chunk)

    chunks.sort(key = lambda c: c["start"])
    return chunks

 
def _build_context(chunks: list[dict], video_title: str = "") -> str:

    lines = [f"{c.get('timestamp', '')} {c['text']}".strip() for c in chunks]
    context = "\n\n".join(lines)
    return f"Video title: {video_title}\n\n{context}" if video_title else context


def _parse_response(raw: str, fallback_title: str) -> ExtractionResult:
    """Strip markdown fences if present, parse JSON, validate with Pydantic."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
 
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON.\nError: {exc}\n"
            f"Response (first 500 chars):\n{raw[:500]}"
        ) from exc
 
    return ExtractionResult(
        title=data.get("title", fallback_title or "Untitled"),
        summary=data.get("summary", ""),
        notes=[Note(**n) for n in data.get("notes", [])],
        action_items=data.get("action_items", []),
        key_concepts=data.get("key_concepts", []),
    )


def extract_notes(store: VectorStore, video_title: str = "") -> ExtractionResult:

    chunks  = _retrieve_chunks(store)
    context = _build_context(chunks, video_title)
    prompt  = _EXTRACTION_PROMPT.format(context=context)
    raw     = generate(prompt, temperature=0.2, max_output_tokens=8192)
    return _parse_response(raw, fallback_title=video_title)