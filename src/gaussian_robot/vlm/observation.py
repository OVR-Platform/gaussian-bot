"""The VLM's per-step input payload (ADR-0005).

An :class:`Observation` is a value object: a list of named image panels plus the
task prompt. It deliberately carries no rendering/coverage logic — that lives in
the builder (:mod:`gaussian_robot.nav.observation`). Keeping the type here means
``vlm`` owns its own input contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Observation:
    """A multi-panel observation for the VLM.

    Attributes
    ----------
    panels:
        Ordered ``(label, image)`` pairs; ``image`` is ``(H, W, 3)`` uint8 RGB.
        Typical order: ``"rgb"``, ``"depth"``, ``"map"`` (ADR-0005).
    prompt:
        The fixed task prompt + one live state line.
    instruction:
        The natural-language goal of a task/VLN episode (ADR-0012), or ``None``
        in coverage mode. Also woven into ``prompt`` for the VLM; carried
        structured here so recorders/evaluators need not re-parse the prompt.
    """

    panels: list[tuple[str, np.ndarray]] = field(default_factory=list)
    prompt: str = ""
    instruction: str | None = None

    def __post_init__(self) -> None:
        for label, img in self.panels:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"panel {label!r} must be (H,W,3) uint8, got {img.shape}")
            if img.dtype != np.uint8:
                raise ValueError(f"panel {label!r} must be uint8, got {img.dtype}")
