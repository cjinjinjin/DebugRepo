import os
import time
import openai
from typing import Optional


_config = None


def set_config(cfg):
    global _config
    _config = cfg


def _get_client():
    return openai.OpenAI(
        api_key=_config.llm_api_key,
        base_url=_config.llm_api_base,
    )


def fill_template(template_path: str, variables: dict) -> str:
    """Read a .txt prompt template and replace {PlaceholderName} with values.

    Uses literal string replacement so LP content containing { or } characters
    does not cause format errors.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    for key, value in variables.items():
        safe_value = str(value) if value else ""
        template = template.replace(f"{{{key}}}", safe_value)
    return template


def call_llm(prompt: str, stream: bool = True, label: str = "") -> str:
    """
    Call LLM with optional streaming output to stdout.
    Retries up to 3 times with exponential backoff on failure.
    """
    client = _get_client()
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(3):
        try:
            if stream:
                collected = []
                response = client.chat.completions.create(
                    model=_config.llm_model,
                    messages=messages,
                    max_tokens=_config.llm_max_tokens,
                    temperature=_config.llm_temperature,
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    collected.append(delta)
                print()
                return "".join(collected)
            else:
                response = client.chat.completions.create(
                    model=_config.llm_model,
                    messages=messages,
                    max_tokens=_config.llm_max_tokens,
                    temperature=_config.llm_temperature,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n[LLM] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"\n[LLM] All retries failed{' for ' + label if label else ''}.")
    return ""


def embed_text(text: str) -> list:
    """Embed a single text string using the configured embedding model."""
    client = _get_client()
    response = client.embeddings.create(
        model=_config.embedding_model,
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list) -> list:
    """Embed a list of texts in one API call."""
    client = _get_client()
    response = client.embeddings.create(
        model=_config.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]
