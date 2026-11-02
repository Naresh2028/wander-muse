"""Try the Reveal on any postcard image and see every step.

    python scripts/reveal_probe.py out/ui/<session>/lofoten-norway.webp
    python scripts/reveal_probe.py .wander/smoke/smoke_3.webp
    python scripts/reveal_probe.py some.webp --cinematic     # no API key needed: tests ffmpeg + OpenCV only

It prints what Muse Spark planned, what SAM 3.1 found for each thing, what it cost, and writes:
    out/probe/layers_debug.jpg   every layer over a checkerboard (do the cut-outs look right?)
    out/probe/reveal.mp4         the animation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from wander.imaging import read_bgr, resize_to_width  # noqa: E402
from wander.reveal.debug import layers_contact_sheet  # noqa: E402
from wander.reveal.layers import single_layer  # noqa: E402
from wander.reveal.motion import CINEMATIC, render_video  # noqa: E402
from wander.reveal.pipeline import VIDEO_WIDTH, bring_to_life, cut_into_layers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/probe"))
    ap.add_argument("--cinematic", action="store_true", help="skip the models; just test the video renderer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    video = args.out / "reveal.mp4"

    if args.cinematic:
        t = time.time()
        work = resize_to_width(read_bgr(args.image), VIDEO_WIDTH)
        render_video(single_layer(work), video, CINEMATIC)
        print(f"Rendered {video} in {time.time() - t:.1f}s (no models used).")
        return 0

    from wander.cache import DiskCache
    from wander.client import MetaClient
    from wander.config import Settings
    from wander.ledger import Ledger

    s = Settings.from_env()
    ledger = Ledger(s.state_dir / "ledger.jsonl", s.budget_usd)
    client = MetaClient(s, ledger, DiskCache(s.state_dir / "cache"))
    work = resize_to_width(read_bgr(args.image), VIDEO_WIDTH)

    print(f"Postcard working size: {work.shape[1]}x{work.shape[0]}, SAM will see max side {s.sam_max_side}px\n")
    notes: list[str] = []
    t = time.time()
    try:
        layers = cut_into_layers(client, work, notes)
        print(f"Cutting took {time.time() - t:.1f}s")
    except Exception as exc:  # noqa: BLE001
        layers = []
        print(f"Cutting FAILED: {type(exc).__name__}: {exc}")
    for n in notes:
        print("  -", n)

    if layers:
        sheet = layers_contact_sheet(layers)
        cv2.imwrite(str(args.out / "layers_debug.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nLayers kept (far -> near): {[l.name for l in layers]}")
        print(f"Wrote {args.out / 'layers_debug.jpg'}  <- open this: do the cut-outs look right?")
    else:
        print("\nNo usable layers, so the video will use the cinematic fallback.")

    t = time.time()
    result = bring_to_life(client, args.image, video)  # re-uses the cached Spark and SAM answers: free
    print(f"\nRendered {result.video} ({result.mode}) in {time.time() - t:.1f}s")
    print(f"Spend so far: {ledger.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
