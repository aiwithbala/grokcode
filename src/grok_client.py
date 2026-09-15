"""xAI Grok client using the OpenAI Python SDK."""

from __future__ import annotations

import os
from collections.abc import Generator, Iterable, Mapping
from typing import Any

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.x.ai/v1"
# Current xAI chat flagship (OpenAI-compatible Chat Completions).
# Override with XAI_MODEL if you prefer another Grok slug.
DEFAULT_MODEL = "grok-4.6"


class MissingAPIKeyError(RuntimeError):
    """Raised when XAI_API_KEY is not configured."""


def _require_api_key() -> str:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "XAI_API_KEY is missing. Copy `.env.example` to `.env`, "
            "add your key from https://console.x.ai/, then restart the app."
        )
    return api_key


def get_client() -> OpenAI:
    """Build an OpenAI SDK client pointed at the xAI API."""
    return OpenAI(
        api_key=_require_api_key(),
        base_url=os.getenv("XAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
    )


def get_model() -> str:
    return os.getenv("XAI_MODEL", DEFAULT_MODEL)


def stream_chat(
    messages: Iterable[Mapping[str, Any]],
) -> Generator[str, None, None]:
    """Stream a chat completion; yields text deltas for Streamlit write_stream."""
    client = get_client()
    stream = client.chat.completions.create(
        model=get_model(),
        messages=list(messages),
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content
