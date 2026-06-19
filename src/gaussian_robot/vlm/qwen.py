"""Concrete VLM client for Qwen3.5-9B served on vLLM (ADR-0001 plug).

Talks to the OpenAI-compatible ``/chat/completions`` endpoint. The
:class:`Observation` panels are JPEG-encoded into ``image_url`` data URLs in the
panel order (rgb, depth, map), matching the prompt. The model runs in thinking
mode by default; its full raw output (including ``<think>``) is kept on the
:class:`Decision` so the dashboard can show its reasoning.

Requires the ``vlm`` extra (``httpx``). Import this module lazily from the UI.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from gaussian_robot.nav.action import Action
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_ACTION_RE = re.compile(r'\{\s*"action"\s*:\s*"([^"]+)"\s*\}', re.IGNORECASE)


def parse_action(raw: str) -> Action:
    """Extract an :class:`Action` from a model response.

    Order: strip ``<think>`` blocks -> find ``{"action": "..."}`` JSON ->
    fall back to any known verb appearing in the text. Raises ``ValueError``
    if nothing parsable is found.
    """
    cleaned = _THINK_RE.sub("", raw).strip()
    m = _JSON_ACTION_RE.search(cleaned)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
    candidates.extend(cleaned.split())
    for verb in candidates:
        v = verb.strip().lower().strip(",.;:`\"'")
        try:
            return Action(v)
        except ValueError:
            continue
    raise ValueError(f"could not parse action from response: {raw!r}")


def jpeg_data_url(image: np.ndarray, *, quality: int = 80) -> str:
    """Encode an ``(H, W, 3)`` uint8 image as a JPEG ``data:`` URL."""
    img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class QwenVLMClient:
    """VLM client over an OpenAI-compatible vLLM endpoint."""

    base_url: str
    model: str
    timeout: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 1024

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def act(self, observation: Observation) -> Decision:
        import httpx  # noqa: PLC0415

        content: list[dict[str, object]] = []
        for _label, image in observation.panels:
            content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(image)}})
        content.append({"type": "text", "text": observation.prompt})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._endpoint(), json=payload)
            resp.raise_for_status()
        data = json.loads(resp.content)
        raw = str(data["choices"][0]["message"]["content"])
        return Decision(action=parse_action(raw), raw_text=raw)
