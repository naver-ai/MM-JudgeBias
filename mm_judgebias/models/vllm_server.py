# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# References for model invocation patterns:
#   - vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI
from PIL import Image

from ..io_utils import pil_to_base64

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _client_kwargs() -> dict[str, str]:
    return {
        "api_key": os.environ.get("VLLM_API_KEY", "EMPTY"),
        "base_url": os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL),
    }


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
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, image),
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    return params


def generate(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    **_: Any,
) -> str:
    client = OpenAI(**_client_kwargs())
    params = _build_params(model, prompt, image, max_tokens, temperature, top_p)
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content or ""


async def generate_async(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
    top_p: float | None = None,
    **_: Any,
) -> str:
    client = AsyncOpenAI(**_client_kwargs())
    params = _build_params(model, prompt, image, max_tokens, temperature, top_p)
    response = await client.chat.completions.create(**params)
    return response.choices[0].message.content or ""
