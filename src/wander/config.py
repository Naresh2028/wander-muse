"""Settings, prices and a tiny .env loader (no extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (never overrides)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Prices:
    """USD prices read from dev.meta.ai on 2026-10-29. The Billing page is the truth;
    this table only drives the local safety cap, so keep it slightly pessimistic."""

    spark_in_per_mtok: float = 1.25  # muse-spark-1.3 (not the -contributor variant)
    spark_out_per_mtok: float = 4.25  # reasoning tokens count as output tokens
    image_each: float = 0.01  # muse-image-1.0, flat per returned image
    sam_per_image: float = 0.0025  # sam-3.1, $2.50 / 1k images
    sam_per_video_frame: float = 0.0002  # sam-3.1, $0.20 / 1k frames
    voice_per_hour: float = 0.18  # muse-voice-transcribe-1.0


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.meta.ai/v1"
    text_model: str = "muse-spark-1.3"
    image_model: str = "muse-image-1.0"
    sam_model: str = "sam-3.1"
    # SAM sees a shrunk copy of the postcard. The official mask decoder is pure Python, so decode
    # time grows with pixel count (about 1.2s per full-frame mask at 768x512, 5.7s at 1536x1024).
    sam_max_side: int = 768
    # Hard cap enforced locally. $150 credit minus a $30 reserve for surprises.
    budget_usd: float = 120.0
    state_dir: Path = Path(".wander")
    # Send the image_generation tool block (size / reasoning_strength / output_format).
    # If the API ever rejects it, set WANDER_IMAGE_TOOL_CONTROLS=0.
    image_tool_controls: bool = True
    prices: Prices = field(default_factory=Prices)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        key = os.environ.get("MODEL_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "MODEL_API_KEY is not set. Copy .env.example to .env and paste the key "
                "from the Meta Model API dashboard (API keys -> Create API key)."
            )
        return cls(
            api_key=key,
            base_url=os.environ.get("WANDER_BASE_URL", cls.base_url),
            text_model=os.environ.get("WANDER_TEXT_MODEL", cls.text_model),
            image_model=os.environ.get("WANDER_IMAGE_MODEL", cls.image_model),
            sam_model=os.environ.get("WANDER_SAM_MODEL", cls.sam_model),
            budget_usd=float(os.environ.get("WANDER_BUDGET_USD", cls.budget_usd)),
            state_dir=Path(os.environ.get("WANDER_STATE_DIR", ".wander")),
            image_tool_controls=os.environ.get("WANDER_IMAGE_TOOL_CONTROLS", "1") != "0",
        )
