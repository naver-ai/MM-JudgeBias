# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

import json
import os

import numpy as np
from PIL import Image, ImageDraw

_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_overlay.json")
LINE_WIDTH = 6

with open(_SPEC_PATH) as _f:
    _SPEC = json.load(_f)


def has_overlay(target_basename):
    return target_basename in _SPEC


def apply_overlay(image, target_basename, line_width=LINE_WIDTH):
    spec = _SPEC.get(target_basename)
    if not spec:
        return image
    im = image.convert("RGB")
    W, H = im.size
    arr = np.asarray(im, np.float32).copy()
    for obj in spec:
        x1, y1, x2, y2 = obj["box"]
        xa, xb = int(max(0, round(x1))), int(min(W, round(x2)))
        ya, yb = int(max(0, round(y1))), int(min(H, round(y2)))
        if xb <= xa or yb <= ya:
            continue
        color = np.array(obj["color"], np.float32)
        a = float(obj["alpha"])
        region = arr[ya:yb, xa:xb]
        region[:] = a * color + (1.0 - a) * region
    out = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(out)
    for obj in spec:
        x1, y1, x2, y2 = obj["box"]
        color = tuple(int(c) for c in obj["color"])
        for t in range(-(line_width // 2), line_width - line_width // 2):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)
    return out
