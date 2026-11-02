import cv2
import numpy as np
import pytest

from conftest import BOAT_BGR, sam_line, sam_record
from wander.ledger import BudgetExceeded
from wander.reveal.layers import LayerSpec, build_layers, clean_concept, single_layer
from wander.reveal.motion import MotionStyle, render_video
from wander.reveal.pipeline import bring_to_life
from wander.reveal.sam import parse_sam_text, union


# ---------------------------------------------------------------- reading SAM's answer
def test_parse_places_a_box_local_mask_back_at_full_size():
    canvas = np.zeros((300, 400), np.uint8)
    canvas[50:120, 100:200] = 1  # a 100x70 rectangle somewhere inside
    canvas[60:70, 110:120] = 0  # with a hole, so decoding is not trivially "all ones"
    parsed = parse_sam_text(sam_line([canvas]), width=400, height=300)
    assert parsed.diagnostics == []
    assert len(parsed.masks) == 1
    assert parsed.masks[0].shape == (300, 400)
    assert np.array_equal(parsed.masks[0], canvas)


def test_parse_handles_several_instances_and_union():
    a = np.zeros((200, 300), np.uint8)
    a[10:40, 10:60] = 1
    b = np.zeros((200, 300), np.uint8)
    b[100:150, 200:280] = 1
    parsed = parse_sam_text(sam_line([a, b]), 300, 200)
    assert len(parsed.masks) == 2
    assert np.array_equal(union(parsed.masks), a | b)


def test_empty_text_means_concept_not_found_not_an_error():
    parsed = parse_sam_text("", 100, 100)
    assert parsed.masks == [] and union(parsed.masks) is None


def test_garbage_text_is_reported_not_raised():
    parsed = parse_sam_text("<0f>this is not a record", 100, 100)
    assert parsed.masks == [] and parsed.diagnostics


def test_clean_concept_accepts_noun_phrases_and_rejects_lists():
    assert clean_concept("Sailboat") == "sailboat"
    assert clean_concept("  Pine   Tree ") == "pine tree"
    for bad in ["boat, tree", "sea and sky", "boat or ship", "a very long phrase here", "", "cat/dog"]:
        assert clean_concept(bad) is None


# ---------------------------------------------------------------- building layers
def _specs(scene):
    return [(LayerSpec(concept="sky", depth=0.0), scene[1]["sky"]),
            (LayerSpec(concept="mountain", depth=0.4), scene[1]["mountain"]),
            (LayerSpec(concept="sailboat", depth=0.9), scene[1]["sailboat"])]


def test_build_layers_orders_far_to_near_with_backdrop_first(scene):
    layers = build_layers(scene[0], _specs(scene))
    assert [l.name for l in layers] == ["backdrop", "sky", "mountain", "sailboat"]
    assert [l.depth for l in layers] == sorted(l.depth for l in layers)
    assert np.all(layers[0].alpha == 1.0)
    for l in layers:
        assert l.rgb.shape == scene[0].shape and l.alpha.shape == scene[0].shape[:2]
        assert 0.0 <= l.alpha.min() and l.alpha.max() <= 1.0


def test_layers_are_filled_behind_nearer_objects(scene):
    img, masks = scene
    layers = {l.name: l for l in build_layers(img, _specs(scene))}
    ys, xs = np.where(masks["sailboat"] > 0)
    cy, cx = int(ys.mean()), int(xs.mean())

    boat = layers["sailboat"]
    assert boat.alpha[cy, cx] > 0.95 and tuple(boat.rgb[cy, cx]) == BOAT_BGR  # the boat itself is intact
    assert boat.alpha[10, 10] < 0.05  # and transparent far away from it

    # The sea/backdrop behind the boat must NOT still contain a red boat (no ghost when it slides)...
    for name in ("backdrop",):
        behind = layers[name].rgb[cy, cx].astype(int)
        assert np.abs(behind - np.array(BOAT_BGR)).sum() > 150, f"{name} still shows the boat"
    # ...and the mountain layer is opaque (filled) there, so a sliding boat never reveals a hole.
    assert layers["mountain"].alpha[cy, cx] > 0.95


def test_tiny_and_giant_masks_are_discarded(scene):
    img, masks = scene
    h, w = img.shape[:2]
    speck = np.zeros((h, w), np.uint8)
    speck[5:9, 5:9] = 1
    everything = np.ones((h, w), np.uint8)
    layers = build_layers(
        img,
        [(LayerSpec(concept="dust", depth=0.5), speck), (LayerSpec(concept="world", depth=0.1), everything)],
    )
    assert layers == []


def test_masks_from_a_smaller_sam_image_are_scaled_to_working_size(scene):
    img, masks = scene
    small = cv2.resize(masks["mountain"], (384, 256), interpolation=cv2.INTER_NEAREST)
    layers = build_layers(img, [(LayerSpec(concept="mountain", depth=0.4), small)])
    assert [l.name for l in layers] == ["backdrop", "mountain"]
    assert layers[1].alpha.shape == img.shape[:2]


# ---------------------------------------------------------------- rendering
def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.int16))
    cap.release()
    return frames


def test_render_video_writes_a_looping_mp4(tmp_path, scene):
    layers = build_layers(scene[0], _specs(scene))
    out = render_video(layers, tmp_path / "v.mp4", MotionStyle(seconds=1.0, fps=12))
    assert out.exists() and not (tmp_path / "v.partial.mp4").exists()
    frames = _read_frames(out)
    assert len(frames) == 12 and frames[0].shape == scene[0].shape
    d = lambda a, b: float(np.abs(frames[a] - frames[b]).mean())
    assert d(0, 3) > 0.05  # things actually move
    # Seamless loop: the jump from the last frame back to the first must be no bigger than a normal
    # frame-to-frame step. (Comparing against a mid-clip frame would be wrong: the sway passes
    # through zero half-way, so that frame legitimately resembles frame 0.)
    steps = [d(i, i + 1) for i in range(11)]
    assert d(11, 0) <= 1.5 * max(steps)


def test_single_layer_cinematic_move_renders(tmp_path, scene):
    out = render_video(single_layer(scene[0]), tmp_path / "c.mp4", MotionStyle(seconds=0.5, fps=10, pan_x=0.02, zoom=0.05))
    assert len(_read_frames(out)) == 5


# ---------------------------------------------------------------- the whole Reveal, with stubs
def test_bring_to_life_cuts_layers_and_renders(scripted, scene_file, tmp_path):
    client, sdk, ledger = scripted
    res = bring_to_life(client, scene_file, tmp_path / "reveal.mp4")
    assert res.mode == "parallax"
    assert res.layers == ["sky", "mountain", "sailboat"]  # far -> near, from Spark's depth values
    assert res.video.exists() and res.video.stat().st_size > 1000
    assert len(sdk.calls["sam"]) == 3
    assert {c["input"][0]["content"][0]["text"] for c in sdk.calls["sam"]} == {"sky", "mountain", "sailboat"}
    assert all(c["metadata"] == {"mask_encoding": "one_bit"} for c in sdk.calls["sam"])
    assert ledger.summary()["sam"] == pytest.approx(3 * 0.0025)
    # Spark was sent the postcard as an image in the user message
    user_msg = sdk.calls["chat"][0]["messages"][1]["content"]
    assert user_msg[1]["type"] == "image_url" and user_msg[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_real_world_path_postcard_is_shrunk_for_sam_and_masks_scaled_back(scripted, tmp_path):
    """The real postcard (1152 wide after resizing) is larger than what SAM receives (768 max side).
    SAM answers in the smaller image's coordinates and the masks must be scaled back up correctly."""
    from conftest import make_scene

    client, sdk, _ = scripted
    big, masks = make_scene(1152, 768)
    path = tmp_path / "big.webp"
    cv2.imwrite(str(path), big)
    sdk.sam_masks = masks  # answered at whatever size SAM is sent

    res = bring_to_life(client, path, tmp_path / "big.mp4")
    assert res.mode == "parallax" and res.layers == ["sky", "mountain", "sailboat"]
    sent = sdk.calls["sam"][0]["input"][0]["content"][1]["image_url"]  # SAM takes the URL as a plain string
    assert sent.startswith("data:image/png;base64,")
    frames = _read_frames(res.video)
    assert frames[0].shape == big.shape  # output stays at full working size, not the SAM size


def test_second_reveal_of_same_postcard_is_free(scripted, scene_file, tmp_path):
    client, sdk, ledger = scripted
    bring_to_life(client, scene_file, tmp_path / "a.mp4")
    spent = ledger.total
    bring_to_life(client, scene_file, tmp_path / "b.mp4")
    assert ledger.total == spent  # Spark plan and SAM answers come from the cache


def test_falls_back_to_cinematic_when_sam_finds_nothing(scripted, scene_file, tmp_path):
    client, sdk, _ = scripted
    sdk.sam_reply = lambda concept: ""
    res = bring_to_life(client, scene_file, tmp_path / "r.mp4")
    assert res.mode == "cinematic" and res.layers == [] and res.video.exists()
    assert any("no match" in n.lower() for n in res.notes)


def test_falls_back_when_spark_plan_is_unusable(scripted, scene_file, tmp_path):
    client, sdk, _ = scripted
    sdk.chat_reply = lambda kw: "{ definitely not the schema"
    res = bring_to_life(client, scene_file, tmp_path / "r.mp4")
    assert res.mode == "cinematic" and res.video.exists()
    assert any("fell back" in n.lower() for n in res.notes)


def test_spend_cap_stops_the_reveal_instead_of_falling_back(make_client, scene_file, tmp_path):
    client, _, _ = make_client(cap=0.0001)
    with pytest.raises(BudgetExceeded):
        bring_to_life(client, scene_file, tmp_path / "r.mp4")
