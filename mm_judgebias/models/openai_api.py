# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# References for model invocation patterns: 
#   - OpenAI python SDK: https://github.com/openai/openai-python
#   - Input message format: https://github.com/openai/openai-python/blob/main/README.md#vision
#   - Synchronous API usage: https://github.com/openai/openai-python/blob/main/README.md#usage
#   - Async API usage: https://github.com/openai/openai-python/blob/main/README.md#async-usage

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI
from PIL import Image

from ..io_utils import pil_to_base64
from .registry import strip_reasoning_suffix

NON_REASONING_MODELS = {"gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"}

API_MODEL_ALIASES = {
    "o4-mini": "o4-mini-2025-04-16",
    "gpt-5.1": "gpt-5.1-2025-11-13",
}


def _resolve_model(model: str) -> tuple[str, bool]:
    base, reasoning = strip_reasoning_suffix(model)
    api_model = API_MODEL_ALIASES.get(base, base)
    return api_model, reasoning


def _build_messages(prompt: str, image: Image.Image | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        data, mime = pil_to_base64(image, format="JPEG")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        )
    return [{"role": "user", "content": content}]


def _build_params(
    model: str,
    prompt: str,
    image: Image.Image | None,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
    verbosity: str | None,
) -> dict[str, Any]:
    api_model, reasoning = _resolve_model(model)
    base_model, _ = strip_reasoning_suffix(model)

    params: dict[str, Any] = {
        "model": api_model,
        "messages": _build_messages(prompt, image),
        "max_completion_tokens": max_tokens,
    }

    if base_model in NON_REASONING_MODELS:
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
    else:
        if reasoning and reasoning_effort is None:
            reasoning_effort = "high"
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            params["verbosity"] = verbosity
    return params


def generate(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    **_: Any,
) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_KEY"], timeout=600)
    params = _build_params(model, prompt, image, max_tokens, temperature, top_p, reasoning_effort, verbosity)
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content or ""


async def generate_async(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    **_: Any,
) -> str:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_KEY"], timeout=600)
    params = _build_params(model, prompt, image, max_tokens, temperature, top_p, reasoning_effort, verbosity)
    response = await client.chat.completions.create(**params)
    return response.choices[0].message.content or ""
