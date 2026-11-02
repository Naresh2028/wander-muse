import pytest
from fastapi.testclient import TestClient

from conftest import make_destinations
from wander.web.app import create_app


@pytest.fixture
def web(make_client, tmp_path):
    client, sdk, ledger = make_client()
    app = create_app(client, ledger, tmp_path / "state", tmp_path / "out")
    return TestClient(app), sdk, ledger, tmp_path


def discover(tc, **extra):
    body = {"vibe": "quiet water and old stone", "days": 4, **extra}
    r = tc.post("/api/discover", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_index_and_static_files_are_served(web):
    tc, *_ = web
    assert "Wander" in tc.get("/").text
    assert tc.get("/static/app.js").status_code == 200
    assert tc.get("/static/style.css").status_code == 200


def test_discover_returns_public_cards_without_painting_prompt(web):
    tc, sdk, ledger, _ = web
    data = discover(tc)
    assert data["round"] == 0 and len(data["destinations"]) == 4
    card = data["destinations"][0]
    assert card["id"] == "r0-0" and "visual_scene" not in card
    assert len(sdk.calls["chat"]) == 1


def test_postcards_paint_and_are_served(web):
    tc, sdk, ledger, _ = web
    data = discover(tc)
    r = tc.post("/api/postcard", json={"session_id": data["session_id"], "dest_id": "r0-1"})
    assert r.status_code == 200
    img = tc.get(r.json()["url"])
    assert img.status_code == 200 and len(img.content) > 0
    # style anchor + one postcard, $0.01 each, plus one Spark call
    assert len(sdk.calls["responses"]) == 2


def test_second_round_sends_feedback_and_skips_seen_places(web):
    tc, sdk, *_ = web
    first = discover(tc)
    sdk.dests = make_destinations(4, prefix="Fresh")
    second = discover(
        tc,
        session_id=first["session_id"],
        liked=["Place 0 <b>"],
        disliked=["Place 3 <b>"],
        note="more water",
    )
    assert second["round"] == 1 and second["destinations"][0]["id"] == "r1-0"
    prompt = sdk.calls["chat"][-1]["messages"][1]["content"]
    assert "Loved: Place 0 <b>" in prompt and "Rejected: Place 3 <b>" in prompt and "more water" in prompt
    assert "Do NOT repeat any place already shown" in prompt and "Place 1 <b>" in prompt


def test_repeated_places_are_filtered_out(web):
    tc, sdk, *_ = web
    first = discover(tc)  # stub keeps returning the same four names
    r = tc.post("/api/discover", json={"session_id": first["session_id"], "note": "again"})
    assert r.status_code == 502 and "already shown" in r.json()["detail"]


def test_choose_and_error_paths(web):
    tc, *_ = web
    data = discover(tc)
    sid = data["session_id"]
    ok = tc.post("/api/choose", json={"session_id": sid, "dest_id": "r0-2"})
    assert ok.status_code == 200 and ok.json()["id"] == "r0-2"
    assert tc.post("/api/choose", json={"session_id": sid, "dest_id": "nope"}).status_code == 404
    assert tc.post("/api/postcard", json={"session_id": "unknown", "dest_id": "r0-0"}).status_code == 404
    assert tc.post("/api/discover", json={"vibe": "  "}).status_code == 422


def test_budget_cap_returns_402(make_client, tmp_path):
    client, _, ledger = make_client(cap=0.0001)
    tc = TestClient(create_app(client, ledger, tmp_path / "state", tmp_path / "out"))
    r = tc.post("/api/discover", json={"vibe": "anything at all"})
    assert r.status_code == 402 and "cap" in r.json()["detail"]


def test_spend_endpoint(web):
    tc, *_ = web
    discover(tc)
    s = tc.get("/api/spend").json()
    assert s["cap"] == 10.0 and s["total"] > 0
