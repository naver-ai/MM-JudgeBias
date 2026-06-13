# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from __future__ import annotations

import importlib
import re
from typing import Any

from PIL import Image

_REASONING_SUFFIXES = ("(reasoning)", "(think)")

PROVIDER_PATTERNS: list[tuple[str, str]] = [
    (r"^(gpt|o\d|chatgpt)", "openai_api"),
    (r"^gemini", "google_api"),
    (r"^claude", "anthropic_api"),
    (r".", "vllm_server"),
]


def strip_reasoning_suffix(model: str) -> tuple[str, bool]:
    for suffix in _REASONING_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)].strip(), True
    return model, False


def resolve_provider(model: str) -> str:
    base, _ = strip_reasoning_suffix(model)
    key = base.lower().split("/")[-1]
    for pattern, provider in PROVIDER_PATTERNS:
        if re.search(pattern, key):
            return provider
    raise ValueError(
        f"Unknown model '{model}'. Add a pattern to PROVIDER_PATTERNS "
        f"in mm_judgebias.models.registry."
    )


MODEL_PROVIDERS = PROVIDER_PATTERNS


def _load_provider(provider: str, async_: bool):
    module = importlib.import_module(f"mm_judgebias.models.{provider}")
    fn_name = "generate_async" if async_ else "generate"
    return getattr(module, fn_name)


def generate(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    **params: Any,
) -> str:
    provider = resolve_provider(model)
    fn = _load_provider(provider, async_=False)
    return fn(model=model, prompt=prompt, image=image, **params)


async def generate_async(
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    **params: Any,
) -> str:
    provider = resolve_provider(model)
    fn = _load_provider(provider, async_=True)
    return await fn(model=model, prompt=prompt, image=image, **params)
