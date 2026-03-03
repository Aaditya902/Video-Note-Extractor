"""
llm/gemini_extractor.py — Extract notes using Google Gemini API (free tier).

Free tier limits (as of 2025):
  - gemini-1.5-flash: 15 RPM, 1500 RPD, 1M token context — perfect for this
  - No credit card required
  - Get your key at: aistudio.google.com

Two-stage approach:
  1. Multi-query retrieval from vector store (RAG)
  2. Pass chronologically ordered chunks to Gemini → structured JSON
"""

import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from models import ExtractionResult, Note
from processing.vector_store import VectorStore

load_dotenv()

# Multiple targeted queries so retriever surfaces diverse chunks
RETRIEVAL_QUERIES = [
    "key ideas concepts explanations definitions",
    "important steps process how to instructions",
    "action items tasks recommendations next steps",
    "examples case studies demonstrations",
    "conclusions summary takeaways results",
]

# Gemini doesn't have a separate system prompt field in the basic API —
# we prepend it as the first part of the user message.
PROMPT_TEMPLATE = """You are an expert note-taking assistant. You receive segments of a video transcript and must extract structured, high-quality notes.
Be concise but complete. Every note should stand alone as a useful piece of information.
Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation.

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
- Timestamps should match the [MM:SS] markers in the context"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    lines = []
    for chunk in chunks:
        ts = chunk.get("timestamp", "")
        lines.append(f"{ts} {chunk['text']}")
    return "\n\n".join(lines)


def extract_notes(store: VectorStore, video_title: str = "") -> ExtractionResult:
    """
    Run multi-query retrieval then call Gemini to extract structured notes.

    Args:
        store:        Populated VectorStore with all video chunks
        video_title:  Optional title hint for the LLM

    Returns:
        Validated ExtractionResult

    Raises:
        EnvironmentError: if GEMINI_API_KEY is not set
        ValueError:       if Gemini returns invalid JSON
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a free key at: https://aistudio.google.com\n"
            "Then add to .env: GEMINI_API_KEY=your_key_here"
        )

    # ── RAG: multi-query retrieval ────────────────────────────────────────────
    seen_texts: set[str] = set()
    all_chunks: list[dict] = []

    for query in RETRIEVAL_QUERIES:
        for chunk in store.query(query, top_k=6):
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_chunks.append(chunk)

    # Sort by position in video for chronological narrative
    all_chunks.sort(key=lambda x: x["start"])

    context = _build_context(all_chunks)
    if video_title:
        context = f"Video title: {video_title}\n\n" + context

    prompt = PROMPT_TEMPLATE.format(context=context)

    # ── Gemini API call ───────────────────────────────────────────────────────
    genai.configure(api_key=api_key)

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    generation_config = genai.types.GenerationConfig(
        temperature=0.2,        # low temp = more consistent structured output
        max_output_tokens=2048,
    )

    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    )

    raw = response.text.strip()

    # Defensive: strip accidental markdown fences Gemini sometimes adds
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # ── Parse + validate ──────────────────────────────────────────────────────
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