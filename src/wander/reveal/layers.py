"""From one flat painting to a stack of layers that can move at different speeds.

The trick behind every "2.5D parallax" effect:
    near things move a lot, far things move a little, and your brain reads that as depth.
To do that we need the picture cut into layers. The cutting is done in three steps:

  1. PLAN    Muse Spark looks at the postcard and names 3-5 things worth cutting out
             ("sky", "mountain", "sailboat") and how far from the viewer each one is.
  2. SEGMENT SAM 3.1 finds the pixels of each named thing (this lives in sam.py + client.py).
  3. BUILD   (this file) turn the masks into layers, and fill in what is hidden behind each
             object so nothing ghosts or tears when the layers slide apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..imaging import read_bgr  # noqa: F401  (re-exported for convenience in scripts)

MAX_LAYERS = 5
MIN_AREA_FRAC = 0.006  # a layer smaller than 0.6% of the picture is not worth moving
MAX_AREA_FRAC = 0.90  # a layer covering nearly everything leaves nothing to separate from


# --------------------------------------------------------------------------- 1. PLAN (Spark)
class LayerSpec(BaseModel):
    concept: str = Field(description="ONE short noun phrase (1-3 words) naming one kind of visible thing.")
    depth: float = Field(description="0.0 = farthest from the viewer (sky), 1.0 = nearest (foreground).")


class LayerPlan(BaseModel):
    layers: list[LayerSpec]


PLAN_SYSTEM = """You prepare a painted travel postcard to be turned into a layered parallax animation.
You are shown the postcard. List the 3 to 5 most visually distinct things in the scene that can be
cut out as separate layers, for example: sky, mountains, sea, a boat, trees, buildings, a person.

Rules:
- Each `concept` is ONE short noun phrase of 1 to 3 words, written the way you would label it for an
  object detector: 'snow mountain', 'sailboat', 'pine tree', 'sky'. One kind of thing per entry.
- Never use commas, 'and', 'or', slashes, or full sentences in a concept.
- Only list things that are clearly visible in the picture. Do not invent objects.
- `depth` is distance from the viewer: 0.0 is the farthest (sky, distant mountains) and 1.0 is the
  nearest (objects in the foreground). Spread the depths out; do not give every layer the same value."""

PLAN_USER = "Here is the postcard. Return the layers."


def clean_concept(text: str) -> str | None:
    """SAM wants ONE short noun phrase. Lists and commands 'often return nothing', so reject them."""
    t = " ".join(text.lower().replace("_", " ").split())
    padded = f" {t} "
    if not t or any(c in t for c in ",;/&+") or " and " in padded or " or " in padded:
        return None
    return t if len(t.split()) <= 3 else None


def plan_layers(client, image_small: np.ndarray) -> list[LayerSpec]:
    """Ask Spark (with the postcard as input) which things to cut out. Returns specs far -> near."""
    plan = client.chat_json(PLAN_SYSTEM, PLAN_USER, LayerPlan, images=[image_small])
    seen: set[str] = set()
    specs: list[LayerSpec] = []
    for s in plan.layers:
        concept = clean_concept(s.concept)
        if concept is None or concept in seen:
            continue
        seen.add(concept)
        specs.append(LayerSpec(concept=concept, depth=float(min(1.0, max(0.0, s.depth)))))
    specs.sort(key=lambda s: s.depth)
    return specs[:MAX_LAYERS]


# --------------------------------------------------------------------------- 3. BUILD
@dataclass
class Layer:
    name: str
    depth: float  # 0 = far, 1 = near. Drives how far the layer moves.
    rgb: np.ndarray  # (H, W, 3) uint8, BGR. Colours of this layer, already filled in behind nearer objects.
    alpha: np.ndarray  # (H, W) float32 in 0..1. 1 = opaque. Soft edges so cut-outs do not look scissored.


def single_layer(image: np.ndarray) -> list[Layer]:
    """The no-cutting fallback: the whole painting as one layer (the renderer then does a slow camera move)."""
    h, w = image.shape[:2]
    return [Layer("scene", 0.0, image, np.ones((h, w), np.float32))]


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(size), _odd(size)))


def _to_working_size(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a 0/1 mask to the working size. Resizing the soft version then thresholding gives
    smoother edges than nearest-neighbour when the mask is enlarged."""
    if mask.shape == (h, w):
        return (mask > 0).astype(np.uint8)
    soft = cv2.resize((mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    return (soft > 127).astype(np.uint8)


def _clean(mask: np.ndarray, k: int) -> np.ndarray:
    """Remove specks (open) and fill pinholes (close). SAM masks are good but never perfect."""
    ker = _ellipse(k)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)


def _inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint over `mask` using the colours around it ("what was probably behind this?").

    Done at low resolution because (a) it is fast and (b) the filled area is only ever glimpsed
    for a few pixels when a layer slides, so detail would be wasted."""
    h, w = image.shape[:2]
    scale = min(1.0, 400 / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, msmall, 3, cv2.INPAINT_TELEA)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)


def build_layers(image: np.ndarray, concept_masks: list[tuple[LayerSpec, np.ndarray]]) -> list[Layer]:
    """Turn (concept, mask) pairs into layers ordered far -> near, with a full-picture backdrop first.

    Returns [] if no mask was usable. Key ideas, in the order the code applies them:

    * EXCLUSIVE CLAIMS. Nearest concept claims its pixels first; a farther concept only gets what is
      left. Without this, 'mountain' and 'tree' would both own the same pixels and fight.
    * BACKDROP. Whatever no concept claimed (plus a fill for the claimed areas) is layer 0, fully opaque.
    * FILLED BEHIND. Every layer is made 'complete' behind the layers in front of it: inside the area
      of nearer objects its colours are inpainted and its alpha is 1. When a near object slides
      sideways, the sliver it uncovers shows plausible filler instead of a hole or a ghost copy.
      That sliver always lies inside the object's original footprint, so only that area is needed.
    """
    h, w = image.shape[:2]
    k = max(3, w // 250)

    # 1) exclusive masks, nearest first
    claimed = np.zeros((h, w), np.uint8)
    kept: list[tuple[LayerSpec, np.ndarray]] = []  # nearest -> farthest
    for spec, raw in sorted(concept_masks, key=lambda sm: sm[0].depth, reverse=True):
        m = _to_working_size(raw, w, h) & (1 - claimed)
        m = _clean(m, k)
        frac = float(m.mean())
        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
            continue
        claimed |= m
        kept.append((spec, m))
    if not kept:
        return []

    far_to_near = list(reversed(kept))
    reach = max(3, w // 200)  # how far past a nearer object's edge we take colours from when inpainting
    ones = np.ones((h, w), np.float32)

    def behind(nearer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rgb, fill_zone) for a layer, given the union mask of everything in front of it."""
        if not nearer.any():
            return image, nearer
        filled = _inpaint(image, cv2.dilate(nearer, _ellipse(2 * reach + 1)))
        zone = cv2.dilate(nearer, _ellipse(5))  # 2px of slack so soft edges blend into the filler
        return np.where(zone[..., None] > 0, filled, image), zone

    # 2) backdrop: everything that is not in a concept layer, with the concept areas filled in
    rgb0, _ = behind(claimed)
    layers = [Layer("backdrop", 0.0, rgb0, ones)]

    # 3) concept layers, far -> near
    sigma = max(1.0, w / 700)
    for i, (spec, own) in enumerate(far_to_near):
        nearer = np.zeros((h, w), np.uint8)
        for _, m in far_to_near[i + 1 :]:
            nearer |= m
        rgb, zone = behind(nearer)
        core = cv2.erode(own, _ellipse(3))  # 1px tighter than SAM's edge so no halo of old background
        alpha = cv2.GaussianBlur(np.maximum(core, zone).astype(np.float32), (0, 0), sigma)
        layers.append(Layer(spec.concept, max(0.05, spec.depth), rgb, np.clip(alpha, 0.0, 1.0)))
    return layers
