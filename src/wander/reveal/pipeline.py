"""Reveal orchestrator: one postcard in, one looping MP4 out.

    postcard.webp --(shrink)--> Spark looks, names layers
                              --> SAM 3.1 cuts each one (in parallel)
                              --> build_layers (fill behind, soft edges)
                              --> render_video (parallax camera) --> reveal.mp4

If anything about the cutting fails (no usable masks, parser problem, odd picture) we do NOT fail the
whole feature: we fall back to a slow cinematic camera move on the uncut painting. The user still
gets a moving postcard, and `RevealResult.notes` records why, so we can fix the cause later.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..imaging import read_bgr, resize_max_side, resize_to_width
from ..ledger import BudgetExceeded
from .layers import Layer, build_layers, plan_layers, single_layer
from .motion import CINEMATIC, PARALLAX, MotionStyle, render_video
from .sam import parse_sam_text, union

log = logging.getLogger("wander.reveal")
VIDEO_WIDTH = 1152  # 1152x768 for a 3:2 postcard: sharp enough for a laptop, light enough to render fast


@dataclass
class RevealResult:
    video: Path
    mode: str  # "parallax" (cut into layers) or "cinematic" (fallback camera move)
    layers: list[str] = field(default_factory=list)  # concept names, far -> near
    notes: list[str] = field(default_factory=list)  # why we fell back, parser warnings, etc.


def cut_into_layers(client, work, notes: list[str]) -> list[Layer]:
    """Plan -> segment -> build. Raises on any problem; the caller decides about the fallback."""
    sam_img = resize_max_side(work, client.sam_max_side)  # small copy: faster mask decoding, fewer tokens
    sh, sw = sam_img.shape[:2]

    plan = plan_layers(client, sam_img)
    if not plan:
        raise ValueError("Spark did not name any usable things to cut out.")
    notes.append("Spark planned: " + ", ".join(f"{s.concept} (depth {s.depth:.2f})" for s in plan))

    def segment(spec):
        # SAM takes ONE concept per request, so this runs once per layer. Threads overlap the waiting.
        parsed = parse_sam_text(client.segment_text(sam_img, spec.concept), sw, sh)
        merged = union(parsed.masks)
        covered = 0.0 if merged is None else 100.0 * float(merged.mean())
        notes.append(f"SAM '{spec.concept}': {len(parsed.masks)} match(es), {covered:.1f}% of the picture")
        notes.extend(f"SAM '{spec.concept}' parser {d}" for d in parsed.diagnostics)
        return spec, merged

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        results = list(pool.map(segment, plan))

    found = [(spec, mask) for spec, mask in results if mask is not None]
    missing = [spec.concept for spec, mask in results if mask is None]
    if missing:
        notes.append(f"SAM found no match for: {', '.join(missing)}")
    return build_layers(work, found)


def bring_to_life(client, image_path: str | Path, out_path: str | Path, *, style: MotionStyle | None = None) -> RevealResult:
    notes: list[str] = []
    work = resize_to_width(read_bgr(image_path), VIDEO_WIDTH)

    layers: list[Layer] = []
    try:
        layers = cut_into_layers(client, work, notes)
    except BudgetExceeded:
        raise  # a spend-cap stop is not a "try something else" situation
    except Exception as exc:  # noqa: BLE001 - any other failure degrades gracefully
        log.warning("Layer cutting failed, using cinematic fallback: %s", exc)
        notes.append(f"Fell back to a cinematic camera move: {type(exc).__name__}: {exc}")

    if len(layers) >= 2:  # backdrop + at least one real layer
        mode, motion = "parallax", style or PARALLAX
    else:
        if not notes:
            notes.append("No usable layers after cleaning; used a cinematic camera move.")
        layers, mode, motion = single_layer(work), "cinematic", style or CINEMATIC

    video = render_video(layers, out_path, motion)
    return RevealResult(video=video, mode=mode, layers=[l.name for l in layers[1:]] if mode == "parallax" else [], notes=notes)
