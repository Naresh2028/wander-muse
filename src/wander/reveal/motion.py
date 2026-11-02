"""Make the layers move and write an MP4.

The camera "sways" on a smooth loop. Each layer follows the camera by an amount that depends on its
depth, which is what creates parallax:

    shift(layer) = camera_motion * factor(depth)         factor goes from 0.1 (far) to 1.0 (near)

Every motion is a sine or cosine of one phase that runs 0 -> 2*pi over the clip, so the last frame
flows straight into the first and the video loops without a visible jump.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .layers import Layer


@dataclass(frozen=True)
class MotionStyle:
    seconds: float = 4.0
    fps: int = 24
    # Every layer is drawn slightly enlarged. Moving a layer by a few pixels then can never slide the
    # picture's edge into view. Margin on each side = (scale - 1) / 2 of the width.
    base_scale: float = 1.07
    pan_x: float = 0.012  # sideways sway, as a fraction of the width, for the NEAREST layer
    pan_y: float = 0.006  # vertical sway, as a fraction of the height
    zoom: float = 0.025  # gentle dolly-in and out


PARALLAX = MotionStyle()
# One flat layer cannot show depth, so use a bigger, slower camera move instead ("Ken Burns").
CINEMATIC = MotionStyle(seconds=5.0, pan_x=0.02, pan_y=0.01, zoom=0.05)


def ffmpeg_exe() -> str:
    """Prefer the ffmpeg that `pip install imageio-ffmpeg` ships, so no system install is needed."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")


def render_video(layers: list[Layer], out_path: str | Path, style: MotionStyle = PARALLAX) -> Path:
    """Composite the layers for every frame and encode an H.264 MP4 (plays in every browser)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = layers[0].rgb.shape[:2]
    if w % 2 or h % 2:
        raise ValueError("Frame size must be even for H.264.")

    # Premultiplied alpha: store colour * alpha with alpha beside it. Warping (resampling) these
    # together blends edges correctly; warping plain RGB would leave dark or bright fringes.
    prem = [np.dstack([l.rgb.astype(np.float32) * l.alpha[..., None], l.alpha]) for l in layers]
    factors = [0.1 + 0.9 * l.depth for l in layers]
    frames = max(2, int(round(style.seconds * style.fps)))

    tmp = out_path.with_name(out_path.stem + ".partial.mp4")  # never serve a half-written file
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(style.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            phase = 2 * math.pi * i / frames
            out = np.zeros((h, w, 3), np.float32)
            for layer_img, f in zip(prem, factors):
                s = style.base_scale + style.zoom * f * (0.5 - 0.5 * math.cos(phase))
                dx = style.pan_x * w * f * math.sin(phase)
                dy = style.pan_y * h * f * math.sin(2 * phase + 0.6)
                # Scale about the centre, then shift: x' = s*x + (1-s)*w/2 + dx
                m = np.float32([[s, 0, (1 - s) * w / 2 + dx], [0, s, (1 - s) * h / 2 + dy]])
                warped = cv2.warpAffine(layer_img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                # "Over" compositing for premultiplied colour: result = layer + below * (1 - layer_alpha)
                out = warped[..., :3] + out * (1.0 - warped[..., 3:4])
            proc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
    except BrokenPipeError:
        pass  # ffmpeg died early; its message is read below
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({code}): {err.strip()[:400]}")
    tmp.replace(out_path)
    return out_path
