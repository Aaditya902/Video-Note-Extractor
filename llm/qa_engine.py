import sys
from pathlib import Path
 
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from processing.vector_store import VectorStore
from llm.gemini_client import generate


QA_PROMPT = (
    "You are a helpful assistant that answers questions ONLY using the video "
    "transcript excerpts provided below.\n"
    "If the answer is not in the excerpts, say exactly: "
    "\"I don't have enough information in the extracted notes to answer that.\"\n"
    "Do not use outside knowledge. Be concise and direct.\n\n"
    "Transcript excerpts:\n{context}\n\n"
    "{history}"
    "User: {question}\n"
    "Assistant:"
)

MAX_HISTORY_TURNS = 3

def format_history(history: list[dict]) -> str:

    if not history:
        return ""
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines) + "\n\n"

def answer(
    question: str,
    store: VectorStore,
    chat_history: list[dict],
) -> str:

    chunks = store.query(question, top_k = 6)
    if not chunks:
        return "I couldn't find relevant information in the extracted notes to answer that."

    context = "\n\n".join(
        f"{c.get('timestamp', '')} {c['text']}".strip()
        for c in chunks
    )
 
    prompt = QA_PROMPT.format(
        context=context,
        history=format_history(chat_history),
        question=question,
    )
 
    return generate(prompt, temperature=0.3, max_output_tokens=1024)
