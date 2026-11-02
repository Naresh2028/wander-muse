import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wander.cache import DiskCache  # noqa: E402
from wander.client import MetaClient  # noqa: E402
from wander.config import Settings  # noqa: E402
from wander.ledger import Ledger  # noqa: E402
from wander.schemas import Destination, DestinationSet  # noqa: E402

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def make_destinations(n=4, prefix="Place"):
    return [
        Destination(
            name=f"{prefix} {i} <b>",
            country="Nowhere",
            tagline="A quiet tagline",
            why_it_fits="Because it fits. Truly.",
            visual_scene="A lantern on a pier at dusk.",
            highlights=["a", "b", "c"],
            best_months="Apr-Jun",
            daily_budget_usd_low=50,
            daily_budget_usd_high=90,
            mood_tags=["calm"],
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------ SAM wire format for tests
def sam_record(mask_full: np.ndarray, index: int = 0) -> str:
    """Encode one full-size 0/1 mask the way SAM 3.1 does: a box plus a box-local, compressed mask.

    Uses Meta's own (private) encoder from `meta-sam-parser`, so the text is byte-for-byte the format
    the parser expects. Skips the test if the library layout ever changes.
    """
    try:
        from meta_sam_parser._mask_codec import _encode_raster
    except ImportError:  # pragma: no cover
        pytest.skip("meta-sam-parser internals changed; cannot build test masks")
    ys, xs = np.where(mask_full > 0)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    sub = (mask_full[y1 : y2 + 1, x1 : x2 + 1] > 0).astype(np.uint8)
    h, w = sub.shape
    big_h, big_w = mask_full.shape
    payload = _encode_raster(sub.flatten().tolist(), w, h)
    return f"{index}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={big_w};h={big_h}|><|mask;x=0;y=0;data={h},{w},{payload}|>"


def sam_line(masks) -> str:
    return "<0f>" + ",".join(sam_record(m, i) for i, m in enumerate(masks))


# ------------------------------------------------------------------ a tiny synthetic postcard
BOAT_BGR = (0, 0, 255)  # pure red so tests can spot it


def make_scene(w=768, h=512):
    """Sky gradient, a mountain, a sea, and a red boat. Returns (image, {name: mask})."""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        img[y, :] = (int(220 - 60 * t), int(170 - 40 * t), int(120 + 40 * t))  # BGR sky tones
    sea_top = int(h * 0.66)
    img[sea_top:, :] = (140, 90, 40)
    mountain = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mountain, [np.array([[int(w * 0.15), sea_top], [int(w * 0.5), int(h * 0.22)], [int(w * 0.85), sea_top]])], 1)
    img[mountain > 0] = (70, 80, 60)
    boat = np.zeros((h, w), np.uint8)
    cv2.rectangle(boat, (int(w * 0.55), int(h * 0.72)), (int(w * 0.68), int(h * 0.80)), 1, -1)
    img[boat > 0] = BOAT_BGR
    sky = np.zeros((h, w), np.uint8)
    sky[: int(h * 0.30), :] = 1
    sky[mountain > 0] = 0
    return img, {"sky": sky, "mountain": mountain, "sailboat": boat}


class _SamStream:
    """Stands in for `client.responses.stream(...)`: iterable events + context manager."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.created")
        half = len(self.text) // 2  # deliver in two chunks like a real stream
        for chunk in (self.text[:half], self.text[half:]):
            if chunk:
                yield SimpleNamespace(type="response.output_text.delta", delta=chunk)
        yield SimpleNamespace(type="response.completed")


class StubSDK:
    """Mimics the OpenAI SDK surface the client uses, with documented response shapes."""

    def __init__(self, dests=None, prompt_tokens=1000, completion_tokens=500, image_ok=True):
        self.dests = dests or make_destinations()
        self.calls = {"chat": [], "responses": [], "sam": []}
        self.prompt_tokens, self.completion_tokens, self.image_ok = prompt_tokens, completion_tokens, image_ok
        self.chat_reply = None  # callable(kwargs) -> JSON string, to script Spark answers
        self.sam_reply = lambda concept: ""  # callable(concept) -> SAM text
        self.sam_masks = None  # or {concept: full-size mask}: answered in the coordinates of the image sent
        self.image_bytes = PNG_1PX  # what Muse Image "paints"; tests may swap in a realistic postcard
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses, stream=self._stream)

    def _chat(self, **kw):
        self.calls["chat"].append(kw)
        content = self.chat_reply(kw) if self.chat_reply else DestinationSet(destinations=self.dests).model_dump_json()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens),
        )

    def _responses(self, **kw):
        self.calls["responses"].append(kw)
        items = [SimpleNamespace(type="reasoning", result=None), SimpleNamespace(type="message", result=None)]
        if self.image_ok:
            items.append(SimpleNamespace(type="image_generation_call", result=base64.b64encode(self.image_bytes).decode()))
        return SimpleNamespace(id="resp_1", status="completed", output=items)

    def _stream(self, **kw):
        self.calls["sam"].append(kw)
        content = kw["input"][0]["content"]
        concept = content[0]["text"]
        if self.sam_masks is not None:
            # Behave like the real SAM: answer in the pixel coordinates of the image it was SENT,
            # which is a shrunk copy of the postcard (see Settings.sam_max_side).
            raw = base64.b64decode(content[1]["image_url"].split(",", 1)[1])
            sent = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            sh, sw = sent.shape[:2]
            mask = self.sam_masks.get(concept)
            text = "" if mask is None else sam_line([cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)])
            return _SamStream(text)
        return _SamStream(self.sam_reply(concept))


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="test", state_dir=tmp_path / "state", budget_usd=10.0)


@pytest.fixture
def make_client(settings, tmp_path):
    def _make(sdk=None, cap=None):
        s = settings if cap is None else Settings(api_key="test", state_dir=tmp_path / "state", budget_usd=cap)
        ledger = Ledger(s.state_dir / "ledger.jsonl", s.budget_usd)
        sdk = sdk or StubSDK()
        return MetaClient(s, ledger, DiskCache(s.state_dir / "cache"), sdk=sdk), sdk, ledger

    return _make


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def scene_file(tmp_path, scene):
    path = tmp_path / "postcard.webp"
    cv2.imwrite(str(path), scene[0])
    return path


@pytest.fixture
def scripted(make_client, scene):
    """A client whose Spark 'sees' sky/mountain/sailboat and whose SAM returns the matching masks."""
    import json

    client, sdk, ledger = make_client()
    _, masks = scene
    sdk.chat_reply = lambda kw: json.dumps(
        {"layers": [{"concept": "sailboat", "depth": 0.9}, {"concept": "sky", "depth": 0.0}, {"concept": "mountain", "depth": 0.4}]}
    )
    sdk.sam_reply = lambda concept: sam_line([masks[concept]]) if concept in masks else ""
    return client, sdk, ledger
