import base64
import sys
from pathlib import Path
from types import SimpleNamespace

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


class StubSDK:
    """Mimics the OpenAI SDK surface the client uses, with documented response shapes."""

    def __init__(self, dests=None, prompt_tokens=1000, completion_tokens=500, image_ok=True):
        self.dests = dests or make_destinations()
        self.calls = {"chat": [], "responses": []}
        self.prompt_tokens, self.completion_tokens, self.image_ok = prompt_tokens, completion_tokens, image_ok
        self.chat_reply = None  # callable(kwargs) -> JSON string, to script Spark answers
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)

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
            items.append(SimpleNamespace(type="image_generation_call", result=base64.b64encode(PNG_1PX).decode()))
        return SimpleNamespace(id="resp_1", status="completed", output=items)


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
