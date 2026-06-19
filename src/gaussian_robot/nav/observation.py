"""Builds the per-step :class:`Observation` (ADR-0005).

Composes a forward render (RGB + depth) with a **body-fixed** top-down coverage
map and the fixed task prompt. The map rotates with the agent each step so empty
regions are always shown relative to the agent's heading (no mental rotation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from gaussian_robot.metrics.coverage import CoverageState, floor_xy
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Camera, Pose
from gaussian_robot.vlm.observation import Observation


def depth_to_uint8(depth: np.ndarray | None) -> np.ndarray:
    """Colormap a depth map to ``(H, W, 3)`` uint8 (near = bright).

    Returns a black panel when ``depth`` is ``None`` or entirely non-finite.
    """
    h = depth.shape[0] if depth is not None else 1
    w = depth.shape[1] if depth is not None else 1
    if depth is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((h, w, 3), dtype=np.uint8)
    lo = float(depth[finite].min())
    hi = float(depth[finite].max())
    span = hi - lo if hi > lo else 1.0
    norm = np.where(finite, 1.0 - (depth - lo) / span, 0.0)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


@dataclass
class ObservationBuilder:
    """Turns a camera + coverage state into an :class:`Observation`.

    Attributes
    ----------
    renderer:
        Anything implementing :class:`~gaussian_robot.render.base.Renderer`.
    up_axis:
        World up axis (drives the floor-plane projection).
    map_size:
        Side length (px) of the square map panel.
    map_span:
        World units spanned across the map. If ``None``, derived from the scene
        AABB diagonal.
    prompt:
        The fixed task prompt prepended to the live state line each step.
    """

    renderer: Renderer
    up_axis: str = "y"
    map_size: int = 512
    map_span: float | None = None
    prompt: str = (
        "You are exploring a 3D scene to propose new camera viewpoints in "
        "under-sampled regions.\n"
        "Panels: [rgb] your current view; [depth] how far surfaces are "
        "(bright = near); [map] a top-down map that rotates with you — blue "
        "dots = already-sampled poses, green line = your trail this walk, red "
        "arrow = you and your heading (up = forward).\n"
        "Steer toward large empty regions, avoid your green trail, and emit "
        '"stop" only when your surroundings look well-covered.\n'
        'Reply with JSON: {"action": <one of '
        "forward|back|turn_left|turn_right|look_up|look_down|stop>}."
    )

    def build(
        self,
        camera: Camera,
        coverage: CoverageState,
        trail: list[Pose],
        *,
        step: int = 0,
        budget: int = 0,
    ) -> tuple[Observation, RenderResult]:
        """Render the view and assemble the observation.

        Returns the :class:`Observation` **and** the underlying
        :class:`RenderResult` (the explorer needs the render to judge
        degeneracy / record confidence).
        """
        result = self.renderer.render(camera)
        depth_panel = depth_to_uint8(result.depth)
        map_panel = self._body_frame_map(coverage, camera.pose, trail)
        state_line = self._state_line(camera.pose, step, budget)
        obs = Observation(
            panels=[
                ("rgb", result.rgb),
                ("depth", depth_panel),
                ("map", map_panel),
            ],
            prompt=self.prompt + "\n" + state_line,
        )
        return obs, result

    def _state_line(self, pose: Pose, step: int, budget: int) -> str:
        step_str = f"{step}/{budget}" if budget > 0 else str(step)
        return (
            f"[state] step {step_str}; pose "
            f"({pose.position[0]:.2f},{pose.position[1]:.2f},{pose.position[2]:.2f})"
        )

    def _body_frame_map(
        self, coverage: CoverageState, current: Pose, trail: list[Pose]
    ) -> np.ndarray:
        size = self.map_size
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        diag = float(np.linalg.norm(coverage.bounds_max - coverage.bounds_min))
        span = self.map_span if self.map_span is not None else max(diag, 1e-9)
        half_span = max(span / 2.0, 1e-9)
        px_per_unit = (size / 2.0) / half_span

        cur_floor = floor_xy(current.position, self.up_axis)[0]
        fwd3 = current.heading(self.up_axis)
        fwd2 = floor_xy(fwd3, self.up_axis)[0]
        n2 = float(np.linalg.norm(fwd2))
        fwd2 = np.array([1.0, 0.0]) if n2 < 1e-9 else fwd2 / n2
        right2 = np.array([-fwd2[1], fwd2[0]])

        def to_pixel(p_floor: np.ndarray) -> tuple[float, float]:
            dp = p_floor - cur_floor
            ahead = float(dp @ fwd2)
            right = float(dp @ right2)
            x = size / 2.0 + right * px_per_unit
            y = size / 2.0 - ahead * px_per_unit
            return (x, y)

        sampled = coverage.floor_positions()
        r_dot = max(3, size // 128)
        for s in sampled:
            x, y = to_pixel(s)
            if -size <= x <= 2 * size and -size <= y <= 2 * size:
                draw.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot), fill=(40, 90, 200))

        if len(trail) >= 2:
            line = [to_pixel(floor_xy(p.position, self.up_axis)[0]) for p in trail]
            draw.line(line, fill=(40, 170, 70), width=max(2, size // 256))

        cx = cy = size / 2.0
        arrow_len = size / 6.0
        draw.line(
            (cx, cy + arrow_len / 2, cx, cy - arrow_len / 2),
            fill=(220, 30, 30),
            width=max(3, size // 100),
        )
        draw.polygon(
            (
                (cx, cy - arrow_len),
                (cx - arrow_len / 3, cy - arrow_len / 2),
                (cx + arrow_len / 3, cy - arrow_len / 2),
            ),
            fill=(220, 30, 30),
        )

        draw.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180))
        return np.asarray(img, dtype=np.uint8)
