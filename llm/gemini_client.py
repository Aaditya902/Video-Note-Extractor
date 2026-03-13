from functools import lru_cache

from google import genai
from google.genai import types

from config import get_gemini_api_key, get_gemini_model


@lru_cache(maxsize = 1)
def get_client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())

def generate(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
) -> str:

    client = get_client()
    model = get_gemini_model()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text.strip()