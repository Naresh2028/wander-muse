"""Thin wrapper over Meta Model API (OpenAI-SDK compatible) with caching, retries and spend control.

Endpoints used (all verified against dev.meta.ai/docs on 2026-10-29):
  * Text + structured JSON : POST /v1/chat/completions, response_format=json_schema, model muse-spark-1.3
  * Images                 : POST /v1/responses, model muse-image-1.0, result is base64 on the
                             `image_generation_call` output item. Reference images go in as `input_image`.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from . import imaging
from .cache import DiskCache
from .config import Settings
from .ledger import Ledger

T = TypeVar("T", bound=BaseModel)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_NAMES = {"APIConnectionError", "APITimeoutError"}


class ImageGenerationError(RuntimeError):
    pass


def _retryable(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in _RETRY_STATUS or type(exc).__name__ in _RETRY_NAMES


def _with_retry(fn, attempts: int = 3, base_delay: float = 1.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if i == attempts - 1 or not _retryable(exc):
                raise
            time.sleep(base_delay * (2**i))


class MetaClient:
    def __init__(self, settings: Settings, ledger: Ledger, cache: DiskCache, sdk=None) -> None:
        self.s = settings
        self.ledger = ledger
        self.cache = cache
        if sdk is None:
            from openai import OpenAI  # imported lazily so tests need no network or SDK

            # The OpenAI SDK does not auto-read MODEL_API_KEY, so it is passed explicitly.
            sdk = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
        self.sdk = sdk

    @property
    def sam_max_side(self) -> int:
        return self.s.sam_max_side

    # ------------------------------------------------------------------ text (Muse Spark)
    def chat_json(
        self,
        system: str,
        user: str,
        model_cls: type[T],
        *,
        images: Sequence[np.ndarray] = (),
        est_usd: float = 0.03,
    ) -> T:
        """Ask Muse Spark for JSON that matches `model_cls`. Cached by (model, prompts, schema, images).

        `images` are BGR arrays (already shrunk by the caller). When given, Spark sees them as part
        of the user message: that is how the Reveal lets Spark *look at* a postcard.
        """
        schema = model_cls.model_json_schema()
        key = self.cache.key("chat", self.s.text_model, system, user, schema, [imaging.digest(i) for i in images])
        hit = self.cache.get_text("chat", key)
        if hit is not None:
            return model_cls.model_validate_json(hit)

        if images:
            # Images are only read from user-role messages. Format per dev.meta.ai/docs/image-understanding.
            user_content: object = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": imaging.data_url(i, "jpeg")}} for i in images
            ]
        else:
            user_content = user

        self.ledger.check(est_usd, f"{self.s.text_model} call")
        resp = _with_retry(
            lambda: self.sdk.chat.completions.create(
                model=self.s.text_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": model_cls.__name__, "schema": schema},
                },
            )
        )
        usage = getattr(resp, "usage", None)
        p, c = (getattr(usage, "prompt_tokens", 0) or 0), (getattr(usage, "completion_tokens", 0) or 0)
        usd = p * self.s.prices.spark_in_per_mtok / 1e6 + c * self.s.prices.spark_out_per_mtok / 1e6
        self.ledger.record("text", self.s.text_model, usd, f"in={p} out={c}")

        content = resp.choices[0].message.content
        try:
            parsed = model_cls.model_validate_json(content)
        except ValidationError as exc:
            raise ValueError(f"Spark returned JSON that does not match {model_cls.__name__}: {content[:400]}") from exc
        self.cache.put_text("chat", key, content)
        return parsed

    # ------------------------------------------------------------ segmentation (SAM 3.1)
    def segment_text(self, image: np.ndarray, concept: str) -> str:
        """Ask SAM 3.1 to find every instance of ONE concept (a short noun phrase) in `image`.

        Returns SAM's raw text: one record per match, each a box then a mask. Parsing lives in
        `reveal/sam.py` so this method only deals with network, cost and cache.
        An empty string is a valid answer: it means the concept is not in the picture.
        """
        key = self.cache.key("sam", self.s.sam_model, imaging.digest(image), concept)
        hit = self.cache.get_text("sam", key)
        if hit is not None:
            return hit

        self.ledger.check(self.s.prices.sam_per_image, "SAM call")
        content = [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": imaging.data_url(image, "png")},
        ]

        def run() -> str:
            parts: list[str] = []
            # Streaming is the documented path for SAM (dev.meta.ai/docs/sam/segmenting).
            # `one_bit` masks are several times smaller on the wire than the default `lossless`.
            with self.sdk.responses.stream(
                model=self.s.sam_model,
                input=[{"type": "message", "role": "user", "content": content}],
                metadata={"mask_encoding": "one_bit"},
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        parts.append(event.delta)
            return "".join(parts)

        text = _with_retry(run)
        self.ledger.record("sam", self.s.sam_model, self.s.prices.sam_per_image, concept)
        self.cache.put_text("sam", key, text)
        return text

    # ----------------------------------------------------------------- images (Muse Image)
    def generate_image(
        self,
        prompt: str,
        *,
        references: Sequence[Path] = (),
        size: str | None = None,
        draft: bool = True,
        output_format: str = "webp",
        tool_controls: bool | None = None,
    ) -> bytes:
        """Generate one image. `draft=True` uses reasoning_strength=low (faster); price is the same.

        `size` is an aspect-ratio hint such as "1536x1024", not exact pixels.
        """
        use_tool = self.s.image_tool_controls if tool_controls is None else tool_controls
        ref_hashes = [self.cache.key(Path(r).read_bytes().hex()[:4096], Path(r).stat().st_size) for r in references]
        key = self.cache.key("img", self.s.image_model, prompt, ref_hashes, size, draft, output_format, use_tool)
        hit = self.cache.get_bytes("img", key, output_format)
        if hit is not None:
            return hit

        self.ledger.check(self.s.prices.image_each, "Muse Image call")

        if references:
            parts: list[dict] = [{"type": "input_text", "text": prompt}]
            for ref in references:
                mime = mimetypes.guess_type(str(ref))[0] or "image/webp"
                b64 = base64.b64encode(Path(ref).read_bytes()).decode()
                parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
            payload: object = [{"role": "user", "content": parts}]
        else:
            payload = prompt

        kwargs: dict = {"model": self.s.image_model, "input": payload}
        if use_tool:
            tool: dict = {"type": "image_generation", "output_format": output_format}
            if size:
                tool["size"] = size
            tool["reasoning_strength"] = "low" if draft else "high"
            kwargs["tools"] = [tool]

        resp = _with_retry(lambda: self.sdk.responses.create(**kwargs))
        b64_result = next(
            (o.result for o in resp.output if getattr(o, "type", None) == "image_generation_call" and o.result),
            None,
        )
        if not b64_result:
            raise ImageGenerationError(
                "Muse Image returned no image (a safety filter or a failed generation). "
                f"Response status: {getattr(resp, 'status', '?')}"
            )
        data = base64.b64decode(b64_result)
        self.ledger.record("image", self.s.image_model, self.s.prices.image_each, f"draft={draft} refs={len(references)}")
        self.cache.put_bytes("img", key, output_format, data)
        return data
