#!/usr/bin/env python

# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Run an MLLM judge over MM-JudgeBias and persist per-sample judgements."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_judgebias.data import HF_DATASET_ID, load_mm_judgebias
from mm_judgebias.judge import JudgeConfig, run_judgement, run_judgement_async


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Judge model identifier")
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable extended thinking (OpenAI reasoning_effort / Gemini thinking_budget / Claude thinking)",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-id", default=HF_DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--bias-types", nargs="+", default=None)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--sync", action="store_true", help="Use the synchronous judge loop")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_mm_judgebias(
        split=args.split,
        repo_id=args.dataset_id,
        cache_dir=args.cache_dir,
        bias_types=args.bias_types,
    )

    cfg = JudgeConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning=args.reasoning,
        seed=args.seed,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        overwrite=args.overwrite,
        images_dir=args.images_dir,
    )

    if args.sync:
        run_judgement(dataset, cfg)
    else:
        asyncio.run(run_judgement_async(dataset, cfg))


if __name__ == "__main__":
    main()
