"""
llm/gemini_client.py — Shared Gemini API client with automatic retry.

Handles 503 UNAVAILABLE (high demand) and 429 RATE_LIMIT errors
automatically with exponential backoff — no user action needed for
temporary server spikes.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
from functools import lru_cache

from google import genai
from google.genai import types

from config import get_gemini_api_key, get_gemini_model


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Return a cached Gemini client. Created once per process."""
    return genai.Client(api_key=get_gemini_api_key())


def get_gemini_model() -> str:
    from config import get_gemini_model as _m
    return _m()


def generate(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    max_retries: int = 3,
) -> str:
    """
    Call Gemini and return response text.

    Automatically retries on 503 (high demand) and 429 (rate limit)
    with exponential backoff: 5s → 15s → 45s.

    Raises:
        RuntimeError: if all retries exhausted
        EnvironmentError: if API key not set
    """
    client     = get_client()
    model      = get_gemini_model()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model    = model,
                contents = prompt,
                config   = types.GenerateContentConfig(
                    temperature       = temperature,
                    max_output_tokens = max_output_tokens,
                ),
            )
            # response.text can be None if Gemini blocked or couldn't process
            if response.text is None:
                # Try to get a reason from the response
                reason = ""
                try:
                    reason = str(response.candidates[0].finish_reason) if response.candidates else ""
                except Exception:
                    pass
                raise ValueError(
                    f"Gemini returned an empty response.\n"
                    f"Finish reason: {reason or 'unknown'}\n"
                    "Possible causes:\n"
                    "  - Video is private, age-restricted, or unavailable\n"
                    "  - Content was blocked by Gemini's safety filters\n"
                    "  - Video is too long for the free tier context window\n"
                    "  - Temporary Gemini API issue — try again"
                )
            return response.text.strip()

        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(x in err_str for x in [
                "503", "unavailable", "429", "rate limit",
                "resource_exhausted", "overloaded", "high demand"
            ])

            if is_retryable and attempt < max_retries - 1:
                wait = 5 * (3 ** attempt)  # 5s, 15s, 45s
                print(f"[Gemini] {e} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                last_error = e
                continue

            # Not retryable or out of retries — raise with clear message
            if is_retryable:
                raise RuntimeError(
                    "Gemini is temporarily unavailable (high demand on free tier).\n"
                    "This is a temporary issue on Google's side — please wait 1-2 minutes and try again.\n"
                    f"Original error: {e}"
                ) from e
            raise

    raise RuntimeError(
        f"Gemini request failed after {max_retries} attempts.\n"
        f"Last error: {last_error}"
    )