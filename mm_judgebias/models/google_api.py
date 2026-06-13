# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# References for model invocation patterns:
#   - Google Gen AI SDK: https://github.com/googleapis/python-genai
#   - Input content and config usage: https://github.com/googleapis/python-genai/blob/main/README.md#using-types
#   - Synchronous API usage: https://github.com/googleapis/python-genai/blob/main/README.md#generate-content
#   - Async API usage: https://github.com/googleapis/python-genai/blob/main/README.md#generate-content-asynchronous-non-streaming

from __future__ import annotations

import io
import os
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from .registry import strip_reasoning_suffix


def _build_content(prompt: str, image: Image.Image | None) -> types.Content:
    parts: list[types.Part] = []
    if image is not None:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        parts.append(
            types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue())
            )
        )
    parts.append(types.Part(text=prompt))
    return types.Content(parts=parts)


def _build_config(
    reasoning: bool,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking_config=types.ThinkingConfig(thinking_budget=-1 if reasoning else 0),
    )


def generate(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    **_: Any,
) -> str:
    api_model, reasoning = strip_reasoning_suffix(model)
    client = genai.Client(api_key=os.environ["GOOGLE_KEY"])
    response = client.models.generate_content(
        model=api_model,
        contents=_build_content(prompt, image),
        config=_build_config(reasoning, max_tokens, temperature, top_p),
    )
    return response.text or ""


async def generate_async(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    **_: Any,
) -> str:
    api_model, reasoning = strip_reasoning_suffix(model)
    client = genai.Client(api_key=os.environ["GOOGLE_KEY"])
    response = await client.aio.models.generate_content(
        model=api_model,
        contents=_build_content(prompt, image),
        config=_build_config(reasoning, max_tokens, temperature, top_p),
    )
    return response.text or ""
