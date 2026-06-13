#!/usr/bin/env python

# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Aggregate persisted judgements into the BD / BC reliability report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_judgebias.evaluate import compute_report, format_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Judge model identifier used at inference time")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def build_model_key(model: str, reasoning: bool) -> str:
    if reasoning and not model.endswith(("(reasoning)", "(think)")):
        model = f"{model}(reasoning)"
    return model


def main() -> None:
    args = parse_args()
    judgement_dir = args.output_dir / "judgements" / f"seed_{args.seed}"
    if not judgement_dir.exists():
        raise SystemExit(f"No judgement directory at {judgement_dir}")

    model_key = build_model_key(args.model, args.reasoning)
    report = compute_report(judgement_dir, model_key=model_key)
    print(format_report(report))

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
