# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from dataclasses import dataclass
from enum import Enum


class BiasCategory(str, Enum):
    INTEGRALITY = "integrality"
    CONGRUITY = "congruity"
    ROBUSTNESS = "robustness"


@dataclass(frozen=True)
class BiasSpec:
    bias_type: str
    category: BiasCategory
    metric: str


BIAS_TAXONOMY: dict[str, BiasSpec] = {
    "text_dominance": BiasSpec("text_dominance", BiasCategory.INTEGRALITY, "bd"),
    "image_dominance": BiasSpec("image_dominance", BiasCategory.INTEGRALITY, "bd"),
    "response_dominance": BiasSpec("response_dominance", BiasCategory.INTEGRALITY, "bd"),
    "instruction_misalignment": BiasSpec("instruction_misalignment", BiasCategory.CONGRUITY, "bd"),
    "image_misalignment": BiasSpec("image_misalignment", BiasCategory.CONGRUITY, "bd"),
    "detail_description": BiasSpec("detail_description", BiasCategory.ROBUSTNESS, "bc"),
    "unnecessary_image": BiasSpec("unnecessary_image", BiasCategory.ROBUSTNESS, "bc"),
    "visual_transformation": BiasSpec("visual_transformation", BiasCategory.ROBUSTNESS, "bc"),
    "texture_insertion": BiasSpec("texture_insertion", BiasCategory.ROBUSTNESS, "bc"),
}

BIAS_TYPES: tuple[str, ...] = tuple(BIAS_TAXONOMY.keys())

CATEGORY_ORDER: tuple[BiasCategory, ...] = (
    BiasCategory.INTEGRALITY,
    BiasCategory.CONGRUITY,
    BiasCategory.ROBUSTNESS,
)
