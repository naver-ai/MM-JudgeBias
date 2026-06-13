# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

import re

_SCORE_MARKER = "### Score:"
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def extract_score(judgement_text: str) -> float | None:
    """Extract the numerical score from a judgment output.

    Returns None when parsing fails; callers are expected to drop such samples.
    """
    if judgement_text is None:
        return None

    text = judgement_text.strip()

    try:
        return float(text)
    except ValueError:
        pass

    if _SCORE_MARKER in text:
        tail = text.split(_SCORE_MARKER)[-1].strip()
        match = _NUMBER_RE.findall(tail)
        if match:
            try:
                return float(match[0])
            except ValueError:
                pass

    match = _NUMBER_RE.findall(text)
    if match:
        try:
            return float(match[-1])
        except ValueError:
            pass

    return None
