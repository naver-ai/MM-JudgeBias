# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from collections.abc import Sequence

from .prompts import MAX_SCORE


def bias_deviation(
    unbiased_scores: Sequence[float],
    biased_scores: Sequence[float],
) -> float:
    """Sensitivity metric for Integrality and Congruity biases.

    BD = E[ max(0, y - y_hat) / (y - 1) | y > 1 ]
    """
    if len(unbiased_scores) != len(biased_scores):
        raise ValueError("unbiased_scores and biased_scores must have equal length")

    values = []
    for y, y_hat in zip(unbiased_scores, biased_scores):
        if y is None or y_hat is None or y <= 1:
            continue
        values.append(max(0.0, y - y_hat) / (y - 1))
    return sum(values) / len(values) if values else 0.0


def bias_conformity(
    unbiased_scores: Sequence[float],
    biased_scores: Sequence[float],
    max_score: int = MAX_SCORE,
) -> float:
    """Stability metric for Robustness biases.

    BC = E[ 1 - |y - y_hat| / max(y - 1, S - y) ]
    """
    if len(unbiased_scores) != len(biased_scores):
        raise ValueError("unbiased_scores and biased_scores must have equal length")

    values = []
    for y, y_hat in zip(unbiased_scores, biased_scores):
        if y is None or y_hat is None:
            continue
        denom = max(y - 1, max_score - y)
        values.append(1.0 - abs(y - y_hat) / denom if denom > 0 else 1.0)
    return sum(values) / len(values) if values else 0.0
