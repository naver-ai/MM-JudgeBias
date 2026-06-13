# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image


def pil_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> tuple[str, str]:
    """Encode a PIL image into base64 with its MIME type."""
    if image.mode != "RGB" and format.upper() in ("JPEG", "JPG"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{'jpeg' if format.upper() == 'JPG' else format.lower()}"
    return data, mime


def image_from_record(field: Any) -> Image.Image | None:
    """Decode an image field as exposed by the HuggingFace dataset loader.

    Supports raw PIL images, {'bytes', 'path'} dicts, and file paths.
    Returns None for missing / text-only samples.
    """
    if field is None:
        return None
    if isinstance(field, Image.Image):
        return field
    if isinstance(field, dict):
        data = field.get("bytes")
        if data:
            return Image.open(io.BytesIO(data))
        path = field.get("path")
        if path:
            return Image.open(path)
        return None
    if isinstance(field, (str, Path)):
        return Image.open(str(field))
    raise TypeError(f"Unsupported image field type: {type(field).__name__}")


def guess_mime_type(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
