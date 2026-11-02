"""Wander pipeline: a vague vibe -> distinct destinations -> postcards in one locked art style,
with a narrowing loop that folds the traveler's reactions into the next round."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Sequence

from .prompts import DESTINATION_SYSTEM, POSTCARD_PROMPT, STYLE_ANCHOR_PROMPT
from .schemas import Destination, DestinationSet, Feedback, TripBrief

POSTCARD_SIZE = "1536x1024"


class WanderClient(Protocol):
    """What the pipeline needs. MetaClient implements it; tests use a stub SDK behind MetaClient."""

    def chat_json(self, system: str, user: str, model_cls, *, est_usd: float = ...): ...

    def generate_image(self, prompt: str, *, references: Sequence[Path] = ..., size=..., draft=..., **kw) -> bytes: ...


def _slug(text: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in text)
    return "-".join(p for p in keep.split("-") if p)[:48] or "place"


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def build_user_prompt(
    brief: TripBrief,
    n: int,
    feedback: Feedback | None = None,
    seen: Sequence[str] = (),
    round_no: int = 0,
) -> str:
    lines = [
        f"Traveler's vibe: {brief.vibe}",
        f"Trip length: {brief.days} days. Budget level: {brief.budget}.",
        f"Travelling from: {brief.travelling_from or 'not stated'}. Companions: {brief.companions or 'not stated'}.",
        f"Avoid: {', '.join(brief.avoid) if brief.avoid else 'nothing stated'}.",
    ]
    if round_no > 0:
        lines.append(f"\nThis is round {round_no + 1}. The traveler reacted to the previous postcards:")
        if feedback and feedback.liked:
            lines.append(f"- Loved: {', '.join(feedback.liked)}")
        if feedback and feedback.disliked:
            lines.append(f"- Rejected: {', '.join(feedback.disliked)}")
        if feedback and feedback.note:
            lines.append(f"- In their words: {feedback.note}")
        lines.append(
            "Move toward what they loved and away from what they rejected, but keep the new places different from "
            "each other. Do NOT repeat any place already shown: " + ", ".join(seen or ["(none)"]) + "."
        )
    lines.append(f"\nReturn exactly {n} destinations.")
    return "\n".join(lines)


def suggest_destinations(
    client: WanderClient,
    brief: TripBrief,
    n: int = 4,
    *,
    feedback: Feedback | None = None,
    seen: Sequence[str] = (),
    round_no: int = 0,
) -> list[Destination]:
    user = build_user_prompt(brief, n, feedback, seen, round_no)
    result = client.chat_json(DESTINATION_SYSTEM, user, DestinationSet)
    seen_norm = {_norm(s) for s in seen}
    fresh = [d for d in result.destinations if _norm(d.name) not in seen_norm][:n]
    if not fresh:
        raise ValueError("Spark only returned places that were already shown. Try again with a different note.")
    if round_no == 0 and len(fresh) < n:
        raise ValueError(f"Spark returned {len(fresh)} destinations, expected {n}.")
    return fresh


def ensure_style_anchor(client: WanderClient, state_dir: Path, *, draft: bool = False) -> Path:
    """The 'Wander look' reference image. Generated once and kept on disk, so re-runs are free.
    Delete `<state_dir>/style_anchor.webp` to repaint it."""
    path = Path(state_dir) / "style_anchor.webp"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.generate_image(STYLE_ANCHOR_PROMPT, size=POSTCARD_SIZE, draft=draft))
    return path


def make_postcard(client: WanderClient, anchor: Path, dest: Destination, out_dir: Path, *, draft: bool) -> Path:
    """Paint one postcard for `dest`, anchored on the shared style reference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = POSTCARD_PROMPT.format(name=dest.name, country=dest.country, visual_scene=dest.visual_scene)
    data = client.generate_image(prompt, references=[anchor], size=POSTCARD_SIZE, draft=draft)
    path = out_dir / f"{_slug(dest.name)}.webp"
    path.write_bytes(data)
    return path


def make_postcards(
    client: WanderClient,
    anchor: Path,
    destinations: Sequence[Destination],
    out_dir: Path,
    *,
    draft: bool,
    workers: int = 4,
) -> dict[str, Path]:
    """One postcard per destination, all anchored on the same style reference."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        paths = pool.map(lambda d: make_postcard(client, anchor, d, out_dir, draft=draft), destinations)
        return {d.name: p for d, p in zip(destinations, paths)}


def run_first_slice(
    client: WanderClient,
    brief: TripBrief,
    *,
    state_dir: Path,
    out_root: Path,
    n: int = 4,
    draft: bool = True,
) -> Path:
    """Command-line slice: returns the path of a generated static index.html."""
    from .render import render_contact_sheet  # local import keeps the module graph simple

    run_dir = Path(out_root) / time.strftime("run-%Y%m%d-%H%M%S")
    dests = suggest_destinations(client, brief, n)
    anchor = ensure_style_anchor(client, state_dir, draft=draft)
    cards = make_postcards(client, anchor, dests, run_dir / "postcards", draft=draft)
    return render_contact_sheet(brief, dests, cards, run_dir)
