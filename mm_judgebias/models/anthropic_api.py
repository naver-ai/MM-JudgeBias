# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# References for model invocation patterns:
#   - Anthropic python SDK: https://github.com/anthropics/anthropic-sdk-python
#   - Input message format: https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/images.py
#   - Synchronous API usage: https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/messages.py
#   - Async API usage: https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/messages_stream.py

from __future__ import annotations

import os
from typing import Any

import anthropic
from PIL import Image

from ..io_utils import pil_to_base64
from .registry import strip_reasoning_suffix

API_MODEL_ALIASES = {
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}

DEFAULT_THINKING_BUDGET = 16384


def _resolve_model(model: str) -> tuple[str, bool]:
    base, reasoning = strip_reasoning_suffix(model)
    api_model = API_MODEL_ALIASES.get(base, base)
    return api_model, reasoning


def _build_content(prompt: str, image: Image.Image | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        data, mime = pil_to_base64(image, format="JPEG")
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def _collect_text(message) -> str:
    return "".join(block.text for block in message.content if hasattr(block, "text"))


def generate(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    **_: Any,
) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"], timeout=300.0)
    api_model, reasoning = _resolve_model(model)

    kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": [{"role": "user", "content": _build_content(prompt, image)}],
    }
    if reasoning:
        kwargs["max_tokens"] = max_tokens + thinking_budget
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    message = client.messages.create(**kwargs)
    return _collect_text(message)


async def generate_async(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    **_: Any,
) -> str:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_KEY"], timeout=300.0)
    api_model, reasoning = _resolve_model(model)

    kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": [{"role": "user", "content": _build_content(prompt, image)}],
    }
    if reasoning:
        kwargs["max_tokens"] = max_tokens + thinking_budget
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    message = await client.messages.create(**kwargs)
    return _collect_text(message)
