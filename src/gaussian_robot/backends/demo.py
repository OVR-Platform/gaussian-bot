"""Demo backends: a synthetic renderer and a scripted VLM.

Used when no real renderer (gsplat) or vLLM endpoint is available, so the
dashboard and tests run anywhere. The fake renderer produces a pose-dependent
image so motion is visible.
"""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import Action
from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation

# A wandering action script that visibly explores a scene.
_DEMO_SCRIPT = [
    "forward",
    "forward",
    "turn_left",
    "forward",
    "look_up",
    "forward",
    "turn_right",
    "forward",
    "forward",
    "turn_right",
    "back",
    "turn_left",
]


class FakeRenderer:
    """Renderer that synthesises a pose-dependent image + valid depth."""

    def __init__(self, *, label: str = "DEMO RENDER") -> None:
        self.label = label
        self.calls = 0

    def render(self, camera: Camera) -> RenderResult:
        self.calls += 1
        h, w = camera.intrinsics.height, camera.intrinsics.width
        pos = camera.pose.position
        x = float(np.clip(pos[0] / 10.0, 0.0, 1.0)) if pos[0] != 0 else 0.2
        z = float(np.clip(pos[2] / 10.0, 0.0, 1.0)) if pos[2] != 0 else 0.3
        r = np.full((h, w), int(x * 255), dtype=np.uint8)
        g = np.full((h, w), int(z * 255), dtype=np.uint8)
        b = np.full((h, w), 80, dtype=np.uint8)
        # add a vertical band that shifts with yaw so turns are visible
        fwd = camera.pose.heading("y")
        phase = (np.arctan2(fwd[0], fwd[2]) / np.pi + 1.0) / 2.0
        col = int(phase * w) % w
        b[:, max(0, col - 6) : col + 6] = 220
        rgb = np.stack([r, g, b], axis=-1)
        depth = np.full((h, w), 5.0, dtype=np.float32)
        return RenderResult(rgb=rgb, camera=camera, depth=depth)


class ScriptedDemoVLM:
    """A VLM that cycles a fixed action script (for demos/tests)."""

    def __init__(self, script: list[str] | None = None, *, label: str = "DEMO VLM") -> None:
        self.label = label
        self._script = script or _DEMO_SCRIPT
        self._idx = 0

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> Decision:
        verb = self._script[self._idx % len(self._script)]
        self._idx += 1
        raw = f'<think>(demo) choosing {verb}</think>\n{{"action": "{verb}"}}'
        return Decision(action=_parse_verb(verb), raw_text=raw)

    def describe(self, observation: Observation) -> str:
        return "Demo scene: a synthetic test environment with colored panels."


def _parse_verb(verb: str) -> Action:
    return Action(verb)
