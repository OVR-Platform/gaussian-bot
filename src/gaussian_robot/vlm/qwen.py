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
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from gaussian_robot.nav.action import Action
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"', re.IGNORECASE)


def parse_action(raw: str) -> tuple[Action, bool]:
    """Extract an :class:`Action` from a model response.

    Order: strip ``<think>`` blocks -> find ``{"action": "..."}`` JSON ->
    fall back to any known verb appearing in the text.

    Returns a ``(action, parse_failed)`` tuple where ``parse_failed`` is
    ``True`` when no known verb was found and the fallback was used.
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
            return Action(v), False
        except ValueError:
            continue
    return Action.TURN_LEFT, True  # fallback: turn rather than drive into a wall


def jpeg_data_url(image: np.ndarray, *, quality: int = 80) -> str:
    """Encode an ``(H, W, 3)`` uint8 image as a JPEG ``data:`` URL."""
    img = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class QwenVLMClient:
    """VLM client over an OpenAI-compatible vLLM endpoint.

    Accumulates multi-turn conversation history within a walk so the VLM
    can see its previous observations and decisions. Call :meth:`reset` at
    the start of each walk.
    """

    base_url: str
    model: str
    timeout: float = 60.0
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    max_history_turns: int = 3
    _history: list[dict[str, object]] = field(default_factory=list)

    def reset(self) -> None:
        self._history.clear()

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @staticmethod
    def _user_message(observation: Observation) -> dict[str, object]:
        """Build a ``user`` message: one image_url per panel plus the prompt text."""
        content: list[dict[str, object]] = [
            {"type": "image_url", "image_url": {"url": jpeg_data_url(image)}}
            for _label, image in observation.panels
        ]
        content.append({"type": "text", "text": observation.prompt})
        return {"role": "user", "content": content}

    @staticmethod
    def _compress_history(messages: list[dict[str, object]]) -> list[dict[str, object]]:
        """Strip images from all but the last user message in history."""
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=None,
        )
        result: list[dict[str, object]] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i != last_user_idx:
                content = msg["content"]
                parts: list[dict[str, object]] = (
                    [c for c in content if c.get("type") == "text"]
                    if isinstance(content, list)
                    else []
                )
                result.append({"role": "user", "content": parts})
            else:
                result.append(msg)
        return result

    def _complete(self, messages: list[dict[str, object]]) -> str:
        """POST ``messages`` to the endpoint and return the raw response text."""
        import httpx  # noqa: PLC0415

        compressed = self._compress_history(messages)
        payload = {
            "model": self.model,
            "messages": compressed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._endpoint(), json=payload)
            resp.raise_for_status()
        data = json.loads(resp.content)
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _save_debug(observation: Observation) -> None:
        from pathlib import Path  # noqa: PLC0415

        dbg = Path("data/vlm_debug")
        dbg.mkdir(parents=True, exist_ok=True)
        for label, image in observation.panels:
            Image.fromarray(image, mode="RGB").save(dbg / f"{label}.png")

    def act(self, observation: Observation) -> Decision:
        self._save_debug(observation)
        self._history.append(self._user_message(observation))
        if self.max_history_turns > 0:
            self._history = self._history[-self.max_history_turns * 2 :]

        raw = self._complete(list(self._history))
        self._history.append({"role": "assistant", "content": raw})
        action, parse_failed = parse_action(raw)
        return Decision(action=action, raw_text=raw, parse_failed=parse_failed)

    def describe(self, observation: Observation) -> str:
        raw = self._complete([self._user_message(observation)])
        return _THINK_RE.sub("", raw).strip()
