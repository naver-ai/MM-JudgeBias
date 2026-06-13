# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm.asyncio import tqdm as tqdm_async

from .data import Sample, iter_samples
from .io_utils import load_json, save_json
from .models import generate, generate_async
from .prompts import EVAL_PROMPT
from .scoring import extract_score


@dataclass
class JudgeConfig:
    model: str
    max_tokens: int = 16384
    temperature: float | None = None
    top_p: float | None = None
    reasoning: bool = False
    seed: int = 1234
    output_dir: Path = Path("outputs")
    max_concurrency: int = 8
    overwrite: bool = False
    images_dir: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.reasoning and not self.model.endswith(("(reasoning)", "(think)")):
            self.model = f"{self.model}(reasoning)"

    @property
    def model_key(self) -> str:
        return self.model

    @property
    def judgement_dir(self) -> Path:
        return self.output_dir / "judgements" / f"seed_{self.seed}"


def _build_params(cfg: JudgeConfig) -> dict[str, Any]:
    params: dict[str, Any] = {"max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        params["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        params["top_p"] = cfg.top_p
    params.update(cfg.extra_params)
    return params


def _sample_record_path(cfg: JudgeConfig, sample_id: int) -> Path:
    return cfg.judgement_dir / f"{sample_id:04d}.json"


def _sample_meta(sample: Sample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "bias_type": sample.bias_type,
        "bias_category": sample.bias_category,
        "task_type": sample.task_type,
        "domain_type": sample.domain_type,
        "difficulty": sample.difficulty,
        "unbiased_source": sample.unbiased_source,
        "biased_source": sample.biased_source,
    }


def _needs_run(record: dict[str, Any], key: str, variant: str, overwrite: bool) -> bool:
    if overwrite:
        return True
    existing = record.get("judgement", {}).get(key, {}).get(variant, "")
    return not (isinstance(existing, str) and existing.strip())


async def _run_one(
    sample: Sample,
    cfg: JudgeConfig,
    params: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        path = _sample_record_path(cfg, sample.id)
        record = load_json(path)
        record.setdefault("meta", _sample_meta(sample))
        judgement = record.setdefault("judgement", {}).setdefault(cfg.model_key, {})

        for variant, instruction, image in (
            ("unbiased", sample.unbiased_instruction, sample.unbiased_image),
            ("biased", sample.biased_instruction, sample.biased_image),
        ):
            if not _needs_run(record, cfg.model_key, variant, cfg.overwrite):
                continue
            prompt = EVAL_PROMPT.format(
                instruction=instruction,
                response=sample.response,
            )
            text = await generate_async(
                model=cfg.model,
                prompt=prompt,
                image=image,
                **params,
            )
            judgement[variant] = text
            judgement[f"{variant}_score"] = extract_score(text)

        save_json(path, record)


async def run_judgement_async(dataset: Dataset, cfg: JudgeConfig) -> None:
    params = _build_params(cfg)
    cfg.judgement_dir.mkdir(parents=True, exist_ok=True)

    samples = list(iter_samples(dataset, images_dir=cfg.images_dir))
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    tasks = [asyncio.create_task(_run_one(s, cfg, params, semaphore)) for s in samples]

    for task in tqdm_async.as_completed(tasks, total=len(tasks), desc=f"Judging [{cfg.model_key}]"):
        await task


def run_judgement(dataset: Dataset, cfg: JudgeConfig) -> None:
    params = _build_params(cfg)
    cfg.judgement_dir.mkdir(parents=True, exist_ok=True)

    for sample in iter_samples(dataset, images_dir=cfg.images_dir):
        path = _sample_record_path(cfg, sample.id)
        record = load_json(path)
        record.setdefault("meta", _sample_meta(sample))
        judgement = record.setdefault("judgement", {}).setdefault(cfg.model_key, {})

        for variant, instruction, image in (
            ("unbiased", sample.unbiased_instruction, sample.unbiased_image),
            ("biased", sample.biased_instruction, sample.biased_image),
        ):
            if not _needs_run(record, cfg.model_key, variant, cfg.overwrite):
                continue
            prompt = EVAL_PROMPT.format(
                instruction=instruction,
                response=sample.response,
            )
            text = generate(model=cfg.model, prompt=prompt, image=image, **params)
            judgement[variant] = text
            judgement[f"{variant}_score"] = extract_score(text)

        save_json(path, record)
