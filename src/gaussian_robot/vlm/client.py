"""VLM client protocol and decision type (ADR-0003, ADR-0004).

Anything that consumes an :class:`Observation` and returns a discrete
:class:`Action` (plus optional raw text) implements :class:`VLMClient`. The
concrete backend — ``Qwen/Qwen3.5-9B`` served on vLLM via an OpenAI-compatible
API — will live in its own module and be injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gaussian_robot.nav.action import Action
from gaussian_robot.vlm.observation import Observation


@dataclass(frozen=True)
class Decision:
    """The VLM's per-step output.

    Attributes
    ----------
    action:
        The chosen :class:`Action` verb. ``STOP`` is handled (demoted) by the
        termination policies, not by the executor.
    raw_text:
        The full decoded model response, kept for logging/reproducibility.
    """

    action: Action
    raw_text: str = ""


@runtime_checkable
class VLMClient(Protocol):
    """Consumes an observation and returns a decision.

    Implementations may accumulate conversation history across calls to
    :meth:`act`. Call :meth:`reset` at the start of each walk to clear it.
    """

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Decision: ...
