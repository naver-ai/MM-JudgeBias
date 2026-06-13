# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from prettytable import PrettyTable

from .bias import BIAS_TAXONOMY, CATEGORY_ORDER, BiasCategory
from .io_utils import load_json
from .metrics import bias_conformity, bias_deviation
from .scoring import extract_score


def _iter_records(judgement_dir: Path):
    for path in sorted(judgement_dir.glob("*.json")):
        yield path, load_json(path)


def _fetch_scores(
    record: dict[str, Any], model_key: str
) -> tuple[float | None, float | None]:
    judgement = record.get("judgement", {}).get(model_key)
    if not judgement:
        return None, None

    unbiased = judgement.get("unbiased_score")
    if unbiased is None and "unbiased" in judgement:
        unbiased = extract_score(judgement["unbiased"])

    biased = judgement.get("biased_score")
    if biased is None and "biased" in judgement:
        biased = extract_score(judgement["biased"])

    return unbiased, biased


def aggregate(
    judgement_dir: Path,
    model_key: str,
) -> dict[str, dict[str, list[float]]]:
    """Collect paired (unbiased, biased) scores grouped by bias type."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"unbiased": [], "biased": []}
    )
    for _, record in _iter_records(judgement_dir):
        bias_type = record.get("meta", {}).get("bias_type")
        if bias_type not in BIAS_TAXONOMY:
            continue
        unbiased, biased = _fetch_scores(record, model_key)
        if unbiased is None or biased is None:
            continue
        grouped[bias_type]["unbiased"].append(float(unbiased))
        grouped[bias_type]["biased"].append(float(biased))
    return grouped


def compute_report(
    judgement_dir: Path,
    model_key: str,
) -> dict[str, Any]:
    grouped = aggregate(judgement_dir, model_key)

    per_bias: dict[str, dict[str, Any]] = {}
    per_category: dict[str, list[float]] = defaultdict(list)

    for bias_type, spec in BIAS_TAXONOMY.items():
        scores = grouped.get(bias_type, {"unbiased": [], "biased": []})
        if not scores["unbiased"]:
            per_bias[bias_type] = {
                "metric": spec.metric,
                "value": None,
                "samples": 0,
                "category": spec.category.value,
            }
            continue

        if spec.metric == "bd":
            value = bias_deviation(scores["unbiased"], scores["biased"])
        else:
            value = bias_conformity(scores["unbiased"], scores["biased"])

        per_bias[bias_type] = {
            "metric": spec.metric,
            "value": value,
            "samples": len(scores["unbiased"]),
            "category": spec.category.value,
        }
        per_category[spec.category.value].append(value)

    category_avg = {
        cat.value: (
            sum(per_category[cat.value]) / len(per_category[cat.value])
            if per_category[cat.value]
            else None
        )
        for cat in CATEGORY_ORDER
    }

    overall_values = [v["value"] for v in per_bias.values() if v["value"] is not None]
    overall_avg = sum(overall_values) / len(overall_values) if overall_values else None

    return {
        "model_key": model_key,
        "per_bias": per_bias,
        "category_avg": category_avg,
        "overall_avg": overall_avg,
    }


def format_report(report: dict[str, Any]) -> str:
    table = PrettyTable()
    table.field_names = ["Category", "Bias Type", "Metric", "Samples", "Score"]
    table.align["Bias Type"] = "l"

    total_samples = 0
    for cat in CATEGORY_ORDER:
        bias_items = [
            (bias_type, spec)
            for bias_type, spec in BIAS_TAXONOMY.items()
            if spec.category == cat
        ]
        cat_samples = 0
        for bias_type, spec in bias_items:
            entry = report["per_bias"][bias_type]
            value = entry["value"]
            cat_samples += entry["samples"]
            table.add_row(
                [
                    cat.value,
                    bias_type,
                    f"{spec.metric.upper()} \u2191",
                    entry["samples"],
                    f"{value:.4f}" if value is not None else "-",
                ]
            )
        total_samples += cat_samples

        cat_val = report["category_avg"].get(cat.value)
        table.add_row(
            [
                "",
                f"({cat.value} avg)",
                "",
                cat_samples,
                f"{cat_val:.4f}" if cat_val is not None else "-",
            ],
            divider=True,
        )

    overall = report["overall_avg"]
    table.add_row(
        [
            "",
            "Overall",
            "",
            total_samples,
            f"{overall:.4f}" if overall is not None else "-",
        ]
    )

    return (
        f"[MM-JudgeBias Report] model={report['model_key']}\n"
        + table.get_string()
    )
