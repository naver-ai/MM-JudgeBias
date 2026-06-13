# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

import os
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import Dataset, load_dataset
from PIL import Image

from .bias import BIAS_TAXONOMY
from .io_utils import image_from_record

HF_DATASET_ID = "naver-ai/MM-JudgeBias"


@dataclass
class Sample:
    id: int
    bias_type: str
    bias_category: str
    task_type: str
    domain_type: str
    difficulty: str
    unbiased_source: str
    biased_source: str
    response: str
    unbiased_instruction: str
    biased_instruction: str
    unbiased_image: Image.Image | None
    biased_image: Image.Image | None

    @classmethod
    def from_record(cls, record: dict[str, Any], images_dir: str | None = None) -> "Sample":
        sample_id = int(record["id"])
        return cls(
            id=sample_id,
            bias_type=record["bias_type"],
            bias_category=record["bias_category"],
            task_type=record["task_type"],
            domain_type=record["domain_type"],
            difficulty=record["difficulty"],
            unbiased_source=record["unbiased_source"],
            biased_source=record["biased_source"],
            response=record["response"],
            unbiased_instruction=record["unbiased_instruction"],
            biased_instruction=record["biased_instruction"],
            unbiased_image=_resolve_image(record.get("unbiased_image"), images_dir, sample_id, "unbiased"),
            biased_image=_resolve_image(record.get("biased_image"), images_dir, sample_id, "biased"),
        )


def _resolve_image(field: Any, images_dir: str | None, sample_id: int, role: str) -> Image.Image | None:
    """Use the dataset image if present; otherwise, for a restricted source whose
    image was reconstructed locally by scripts/prepare_dataset.py, load it from images_dir."""
    image = image_from_record(field)
    if image is not None or not images_dir:
        return image
    for name in (f"{sample_id:04d}_{role}.jpg", f"{sample_id:04d}.jpg"):
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return Image.open(path)
    return None


def load_mm_judgebias(
    split: str = "test",
    repo_id: str = HF_DATASET_ID,
    cache_dir: str | None = None,
    bias_types: list[str] | None = None,
) -> Dataset:
    """Load MM-JudgeBias from the HuggingFace Hub.

    Restricted-source images ship as null cells; reconstruct them with
    ``scripts/prepare_dataset.py`` and pass ``images_dir`` to ``iter_samples`` (see
    prepare/DATA_PREPARATION.md).
    """
    dataset = load_dataset(repo_id, split=split, cache_dir=cache_dir)

    if bias_types is not None:
        unknown = set(bias_types) - set(BIAS_TAXONOMY.keys())
        if unknown:
            raise ValueError(f"Unknown bias types: {sorted(unknown)}")
        dataset = dataset.filter(lambda ex: ex["bias_type"] in bias_types)

    return dataset


def iter_samples(dataset: Dataset, images_dir: str | None = None) -> Iterator[Sample]:
    for record in dataset:
        yield Sample.from_record(record, images_dir=images_dir)
