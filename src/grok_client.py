"""xAI Grok client using the OpenAI Python SDK (text + multimodal)."""

from __future__ import annotations

import os
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.x.ai/v1"
# Current xAI chat flagship (OpenAI-compatible Chat Completions).
# Override with XAI_MODEL if you prefer another Grok slug.
DEFAULT_MODEL = "grok-4.6"
# Vision-capable default when the request includes images.
# grok-4.6 supports image understanding on the xAI API (see docs).
# XAI_MODEL still wins when set.
DEFAULT_VISION_MODEL = "grok-4.6"


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


def get_model(*, has_images: bool = False) -> str:
    """Return the model slug. Uses vision default when images are present unless overridden."""
    override = os.getenv("XAI_MODEL", "").strip()
    if override:
        return override
    if has_images:
        return DEFAULT_VISION_MODEL
    return DEFAULT_MODEL


def build_image_part(data_url: str, *, detail: str = "auto") -> dict[str, Any]:
    """OpenAI-compatible image_url content part for xAI chat completions."""
    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
            "detail": detail,
        },
    }


def build_text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def build_multimodal_user_message(
    instruction: str,
    *,
    document_context: str = "",
    youtube_context: str = "",
    image_data_urls: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Build a user message for chat completions.

    Text-only → plain string content.
    With images → content parts list (text + image_url) as expected by xAI.
    """
    sections: list[str] = []
    if document_context.strip():
        sections.append(document_context.strip())
    if youtube_context.strip():
        sections.append(youtube_context.strip())
    instruction = instruction.strip()
    if instruction:
        sections.append(f"### User instruction\n\n{instruction}")
    elif not image_data_urls:
        sections.append("### User instruction\n\n(Please review the attached material.)")

    text_blob = "\n\n".join(sections).strip()
    images = [u for u in (image_data_urls or []) if u]
    if not images:
        return {"role": "user", "content": text_blob or instruction}

    parts: list[dict[str, Any]] = []
    if text_blob:
        parts.append(build_text_part(text_blob))
    for url in images:
        parts.append(build_image_part(url))
    return {"role": "user", "content": parts}


def stream_chat(
    messages: Iterable[Mapping[str, Any]],
    *,
    has_images: bool = False,
) -> Generator[str, None, None]:
    """Stream a chat completion; yields text deltas for Streamlit write_stream."""
    client = get_client()
    stream = client.chat.completions.create(
        model=get_model(has_images=has_images),
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
