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


def _script_reveal(sdk, scene):
    """Make the stubbed Muse Image paint the synthetic scene, Spark plan layers, SAM return masks."""
    import json

    import cv2

    from conftest import sam_line
    from wander.schemas import DestinationSet

    plan = json.dumps({"layers": [{"concept": "sky", "depth": 0.0}, {"concept": "mountain", "depth": 0.4}, {"concept": "sailboat", "depth": 0.9}]})
    dests = DestinationSet(destinations=sdk.dests).model_dump_json()
    sdk.chat_reply = lambda kw: plan if kw["response_format"]["json_schema"]["name"] == "LayerPlan" else dests
    sdk.sam_reply = lambda concept: sam_line([scene[1][concept]])
    sdk.image_bytes = cv2.imencode(".png", scene[0])[1].tobytes()


def test_reveal_endpoint_animates_a_painted_postcard(web, scene):
    tc, sdk, ledger, _ = web
    _script_reveal(sdk, scene)
    sid = discover(tc)["session_id"]
    assert tc.post("/api/postcard", json={"session_id": sid, "dest_id": "r0-0"}).status_code == 200

    r = tc.post("/api/reveal", json={"session_id": sid, "dest_id": "r0-0"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "parallax" and data["layers"] == ["sky", "mountain", "sailboat"]
    video = tc.get(data["video_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4" and len(video.content) > 1000

    # Asking again (re-opening the postcard) is instant and costs nothing more
    before = (len(sdk.calls["sam"]), ledger.total)
    again = tc.post("/api/reveal", json={"session_id": sid, "dest_id": "r0-0"})
    assert again.json() == data and (len(sdk.calls["sam"]), ledger.total) == before


def test_reveal_before_the_postcard_is_painted_is_a_clear_409(web):
    tc, *_ = web
    sid = discover(tc)["session_id"]
    r = tc.post("/api/reveal", json={"session_id": sid, "dest_id": "r0-1"})
    assert r.status_code == 409 and "painting" in r.json()["detail"]
    assert tc.post("/api/reveal", json={"session_id": sid, "dest_id": "nope"}).status_code == 404


def test_reveal_degrades_to_a_cinematic_move_when_sam_finds_nothing(web, scene):
    tc, sdk, *_ = web
    _script_reveal(sdk, scene)
    sdk.sam_reply = lambda concept: ""
    sid = discover(tc)["session_id"]
    tc.post("/api/postcard", json={"session_id": sid, "dest_id": "r0-2"})
    r = tc.post("/api/reveal", json={"session_id": sid, "dest_id": "r0-2"})
    assert r.status_code == 200 and r.json()["mode"] == "cinematic" and r.json()["layers"] == []


def test_spend_endpoint(web):
    tc, *_ = web
    discover(tc)
    s = tc.get("/api/spend").json()
    assert s["cap"] == 10.0 and s["total"] > 0
