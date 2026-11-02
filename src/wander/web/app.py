"""FastAPI app for the Wander postcard deck.

Flow the browser follows:
  POST /api/discover  -> Spark picks destinations (and the style anchor is ensured, in parallel)
  POST /api/postcard  -> one call per card, fired concurrently, so cards fill in as they finish
  POST /api/choose    -> the traveler picks a place
  POST /api/reveal    -> the chosen postcard comes alive (Spark looks, SAM cuts, we render an MP4)
  GET  /api/spend     -> local ledger summary for the on-screen spend meter
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..client import ImageGenerationError
from ..ledger import BudgetExceeded, Ledger
from ..pipeline import WanderClient, ensure_style_anchor, make_postcard, suggest_destinations
from ..reveal.pipeline import bring_to_life
from ..schemas import Destination, Feedback, TripBrief

log = logging.getLogger("wander.web")
STATIC_DIR = Path(__file__).parent / "static"
N_DESTINATIONS = 4


class DiscoverRequest(BaseModel):
    session_id: str | None = None
    vibe: str | None = Field(default=None, max_length=600)
    days: int = Field(default=5, ge=1, le=60)
    budget: Literal["shoestring", "comfortable", "splurge"] = "comfortable"
    avoid: list[str] = Field(default_factory=list, max_length=10)
    liked: list[str] = Field(default_factory=list, max_length=8)
    disliked: list[str] = Field(default_factory=list, max_length=8)
    note: str | None = Field(default=None, max_length=400)


class PostcardRequest(BaseModel):
    session_id: str
    dest_id: str
    final: bool = False


class ChooseRequest(BaseModel):
    session_id: str
    dest_id: str


class RevealRequest(BaseModel):
    session_id: str
    dest_id: str


@dataclass
class Session:
    id: str
    brief: TripBrief
    round_no: int = 0
    seen: list[str] = field(default_factory=list)
    dests: dict[str, Destination] = field(default_factory=dict)
    images: dict[str, Path] = field(default_factory=dict)  # dest_id -> painted postcard file on disk
    reveals: dict[str, dict] = field(default_factory=dict)  # dest_id -> finished reveal (so re-opening is instant)
    chosen: str | None = None


def _public(dest_id: str, d: Destination) -> dict:
    """What the browser may see. `visual_scene` stays server-side (it is only a painting prompt)."""
    out = d.model_dump(exclude={"visual_scene"})
    out["id"] = dest_id
    return out


def create_app(client: WanderClient, ledger: Ledger, state_dir: Path, out_dir: Path) -> FastAPI:
    app = FastAPI(title="Wander", docs_url=None, redoc_url=None)
    sessions: dict[str, Session] = {}
    lock = threading.Lock()
    media_dir = Path(out_dir) / "ui"
    media_dir.mkdir(parents=True, exist_ok=True)

    def get_session(session_id: str) -> Session:
        sess = sessions.get(session_id)
        if sess is None:
            raise HTTPException(404, "That session has expired. Start a new one.")
        return sess

    def fail(exc: Exception) -> HTTPException:
        if isinstance(exc, BudgetExceeded):
            return HTTPException(402, str(exc))
        if isinstance(exc, ImageGenerationError):
            return HTTPException(502, "That postcard wouldn't paint. Try again.")
        if isinstance(exc, ValueError):
            return HTTPException(502, str(exc))
        log.exception("Upstream failure")
        return HTTPException(502, "Something went wrong talking to the Meta Model API. Try again.")

    @app.post("/api/discover")
    def discover(req: DiscoverRequest) -> dict:
        with lock:
            sess = sessions.get(req.session_id) if req.session_id else None
            if sess is None:
                if not req.vibe or len(req.vibe.strip()) < 3:
                    raise HTTPException(422, "Tell me a little about the feeling you are after.")
                sess = Session(
                    id=uuid.uuid4().hex,
                    brief=TripBrief(vibe=req.vibe.strip(), days=req.days, budget=req.budget, avoid=req.avoid),
                )
                sessions[sess.id] = sess
            feedback = Feedback(liked=req.liked, disliked=req.disliked, note=req.note) if sess.round_no else None
            round_no, seen = sess.round_no, list(sess.seen)

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_dests = pool.submit(
                    suggest_destinations, client, sess.brief, N_DESTINATIONS,
                    feedback=feedback, seen=seen, round_no=round_no,
                )
                f_anchor = pool.submit(ensure_style_anchor, client, state_dir, draft=False)
                dests, _anchor = f_dests.result(), f_anchor.result()
        except Exception as exc:  # noqa: BLE001
            raise fail(exc) from exc

        with lock:
            cards = []
            for i, d in enumerate(dests):
                dest_id = f"r{round_no}-{i}"
                sess.dests[dest_id] = d
                sess.seen.append(d.name)
                cards.append(_public(dest_id, d))
            sess.round_no = round_no + 1
        return {"session_id": sess.id, "round": round_no, "destinations": cards}

    @app.post("/api/postcard")
    def postcard(req: PostcardRequest) -> dict:
        sess = get_session(req.session_id)
        dest = sess.dests.get(req.dest_id)
        if dest is None:
            raise HTTPException(404, "Unknown destination.")
        anchor = Path(state_dir) / "style_anchor.webp"
        if not anchor.is_file():
            raise HTTPException(409, "The style reference is not ready yet.")
        try:
            path = make_postcard(client, anchor, dest, media_dir / sess.id, draft=not req.final)
        except Exception as exc:  # noqa: BLE001
            raise fail(exc) from exc
        sess.images[req.dest_id] = path  # remembered so /api/reveal can find the file later
        return {"url": f"/media/{sess.id}/{path.name}"}

    @app.post("/api/choose")
    def choose(req: ChooseRequest) -> dict:
        sess = get_session(req.session_id)
        dest = sess.dests.get(req.dest_id)
        if dest is None:
            raise HTTPException(404, "Unknown destination.")
        sess.chosen = req.dest_id
        return _public(req.dest_id, dest)

    reveal_locks: dict[tuple[str, str], threading.Lock] = {}

    @app.post("/api/reveal")
    def reveal(req: RevealRequest) -> dict:
        """Animate the chosen postcard. Slow (several seconds), so the browser shows a status line.

        FastAPI runs plain `def` endpoints in a worker thread, so a long render here does not freeze
        other requests. A per-postcard lock stops a double-click from rendering the same video twice.
        """
        sess = get_session(req.session_id)
        if req.dest_id not in sess.dests:
            raise HTTPException(404, "Unknown destination.")
        image = sess.images.get(req.dest_id)
        if image is None or not image.is_file():
            raise HTTPException(409, "Wait for the postcard to finish painting first.")

        with lock:
            key_lock = reveal_locks.setdefault((sess.id, req.dest_id), threading.Lock())
        with key_lock:
            done = sess.reveals.get(req.dest_id)
            if done is None:
                out = media_dir / sess.id / f"{image.stem}-reveal.mp4"
                try:
                    result = bring_to_life(client, image, out)
                except Exception as exc:  # noqa: BLE001
                    raise fail(exc) from exc
                for note in result.notes:  # internals stay in the server log, not in the browser
                    log.info("reveal %s: %s", req.dest_id, note)
                done = {"video_url": f"/media/{sess.id}/{out.name}", "mode": result.mode, "layers": result.layers}
                sess.reveals[req.dest_id] = done
        return done

    @app.get("/api/spend")
    def spend() -> dict:
        s = ledger.summary()
        return {"total": s.get("total", 0.0), "cap": s.get("cap", ledger.cap_usd)}

    app.mount("/media", StaticFiles(directory=media_dir), name="media")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
