import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

import sys
from pathlib import Path as _root_path
sys.path.insert(0, str(_root_path(__file__).resolve().parent.parent))

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

PROMPT_TEMPLATE = """You are an expert note-taking assistant. You receive segments of a video transcript and must extract structured, high-quality notes.
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


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    lines = []
    for chunk in chunks:
        ts = chunk.get("timestamp", "")
        lines.append(f"{ts} {chunk['text']}")
    return "\n\n".join(lines)


def extract_notes(store: VectorStore, video_title: str = "") -> ExtractionResult:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com\n"
            "Then add to .env: GEMINI_API_KEY=your_key_here"
        )

    seen_texts: set[str] = set()
    all_chunks: list[dict] = []

    for query in RETRIEVAL_QUERIES:
        for chunk in store.query(query, top_k=6):
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x["start"])

    context = _build_context(all_chunks)
    if video_title:
        context = f"Video title: {video_title}\n\n" + context

    prompt = PROMPT_TEMPLATE.format(context=context)

    client = genai.Client(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,   # raised from 2048 — prevents truncated JSON
        ),
    )

    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Parse + validate 
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON.\nError: {e}\nRaw response:\n{raw[:500]}"
        )

    result = ExtractionResult(
        title=data.get("title", video_title or "Untitled"),
        summary=data.get("summary", ""),
        notes=[Note(**n) for n in data.get("notes", [])],
        action_items=data.get("action_items", []),
        key_concepts=data.get("key_concepts", []),
    )
    return result