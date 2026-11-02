"""Small image helpers shared by the Spark-vision call, the SAM call and the Reveal renderer.

Why this file exists: OpenCV stores images as NumPy arrays in BGR order (blue, green, red), and
every place that sends a picture to the API needs the same few steps (read, shrink, encode,
wrap as a data URL). Keeping them here means the rest of the code never repeats them.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    """Load an image file (webp/png/jpg) as a BGR uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read an image from {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longer side is at most `max_side` (never enlarges)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    """Resize to `width` keeping the aspect ratio. Both sides are forced even, which H.264 video needs."""
    h, w = img.shape[:2]
    width = max(2, min(width, w) // 2 * 2)
    height = max(2, round(h * width / w) // 2 * 2)
    if (width, height) == (w, h):
        return img
    interp = cv2.INTER_AREA if width < w else cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=interp)


def encode(img: np.ndarray, fmt: str = "jpeg", quality: int = 90) -> bytes:
    """Encode to jpeg/png bytes."""
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Image encoding failed")
    return buf.tobytes()


def data_url(img: np.ndarray, fmt: str = "jpeg", quality: int = 90) -> str:
    """`data:image/...;base64,...` string, the form both Muse Spark and SAM accept for inline images."""
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(encode(img, fmt, quality)).decode()}"


def digest(img: np.ndarray) -> str:
    """Stable short fingerprint of the pixels, used in cache keys so identical images hit the cache."""
    return hashlib.sha256(img.tobytes()).hexdigest()[:24]
