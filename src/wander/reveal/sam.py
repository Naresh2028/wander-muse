"""Turn SAM 3.1's text answer into plain NumPy masks.

What SAM sends back (see dev.meta.ai/docs/sam/reading-segmentation) is one line of text per image:

    <0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600|><|mask;x=0;y=0;data=122,376,!....|>,1<|box...

Each match is a *box* plus a *mask*, and the mask only covers the inside of its box ("box-local").
So there are two jobs:
  1. decode the compressed mask into 0/1 pixels  -> done by Meta's official `meta-sam-parser`
  2. paste each box-local mask back at the right spot of a full-size canvas -> done here

We use the official parser instead of writing our own decoder because the compression format is
custom (a range coder over base85 text) and the docs do not describe it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format


@dataclass
class SamParse:
    masks: list[np.ndarray] = field(default_factory=list)  # each (H, W) uint8 of 0/1, full image size
    diagnostics: list[str] = field(default_factory=list)  # parser warnings, kept for debugging


def parse_sam_text(text: str, width: int, height: int) -> SamParse:
    """Decode every mask in `text` into a full-size (height x width) 0/1 array.

    `width`/`height` are the size of the image that was SENT to SAM, because the box coordinates
    are in that image's pixels. Empty text is a normal result (concept not found) and gives no masks.
    """
    out = SamParse()
    if not text.strip():
        return out

    # The parser is built for streaming; feeding it the whole text at once and then calling
    # finish() is the supported way to read a complete response.
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result  # finish() wraps (events, result); we want the result
    out.diagnostics = [f"{d.severity}: {d.code}: {d.message}" for d in result.diagnostics]

    for rec in result.records:
        if getattr(rec, "kind", None) != "mask":
            continue  # box-only records carry no pixels
        mask = rec.mask
        raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(mask.height, mask.width)

        # `bounds` is the half-open box (left, top, right, bottom) in source-image pixels.
        b = rec.bounds
        x0, y0 = int(round(b.left)), int(round(b.top))
        bw, bh = max(1, int(round(b.right - b.left))), max(1, int(round(b.bottom - b.top)))
        if (mask.width, mask.height) != (bw, bh):
            # Defensive: if the mask raster and its box ever disagree, stretch the mask to the box.
            raster = cv2.resize(raster, (bw, bh), interpolation=cv2.INTER_NEAREST)

        full = np.zeros((height, width), dtype=np.uint8)
        x1, y1 = min(width, x0 + bw), min(height, y0 + bh)
        if x1 > x0 and y1 > y0:
            full[y0:y1, x0:x1] = raster[: y1 - y0, : x1 - x0]
        out.masks.append(full)
    return out


def union(masks: list[np.ndarray]) -> np.ndarray | None:
    """Merge all instances of one concept (several trees, two boats) into a single mask."""
    if not masks:
        return None
    merged = masks[0].copy()
    for m in masks[1:]:
        merged |= m
    return merged
