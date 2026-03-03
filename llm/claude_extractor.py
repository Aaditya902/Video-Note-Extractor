import json
import os

import anthropic
from dotenv import load_dotenv

from models import ExtractionResult, Note
from processing.vector_store import VectorStore

load_dotenv()


RETRIEVAL_QUERIES = [
    "key ideas concepts explanations definitions",
    "important steps process how to instructions",
    "action items tasks recommendations next steps",
    "examples case studies demonstrations",
    "conclusions summary takeaways results",
]

SYSTEM_PROMPT = """You are an expert note-taking assistant. You receive segments of a video transcript and must extract structured, high-quality notes.

Be concise but complete. Every note should stand alone as a useful piece of information.
Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation."""

EXTRACTION_PROMPT = """Extract structured notes from these transcript segments:

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
- Timestamps should match the [MM:SS] markers in the context"""


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        ts = chunk.get("timestamp", "")
        lines.append(f"{ts} {chunk['text']}")
    return "\n\n".join(lines)


def extract_notes(store: VectorStore, video_title: str = "") -> ExtractionResult:

    seen_texts: set[str] = set()
    all_chunks: list[dict] = []

    for query in RETRIEVAL_QUERIES:
        for chunk in store.query(query, top_k=6):
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_chunks.append(chunk)

    # Re-sort by video position for chronological context
    all_chunks.sort(key=lambda x: x["start"])

    context = _build_context(all_chunks)
    if video_title:
        context = f"Video title: {video_title}\n\n" + context

    prompt = EXTRACTION_PROMPT.format(context=context)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Defensive: strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    # Validate and coerce with Pydantic
    result = ExtractionResult(
        title=data.get("title", video_title or "Untitled"),
        summary=data.get("summary", ""),
        notes=[Note(**n) for n in data.get("notes", [])],
        action_items=data.get("action_items", []),
        key_concepts=data.get("key_concepts", []),
    )
    return result