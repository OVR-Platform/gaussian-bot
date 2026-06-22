"""Builds the per-step :class:`Observation` (ADR-0005).

Composes a forward render (RGB + depth) with a **body-fixed** top-down coverage
map and the fixed task prompt. The map rotates with the agent each step so empty
regions are always shown relative to the agent's heading (no mental rotation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from gaussian_robot.depth.estimator import DepthEstimator
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
    task: str = ""
    depth_estimator: DepthEstimator | None = None
    prompt: str = (
        "You are a robot exploring a 3D scene to find gaps in the reconstruction.\n"
        "\n"
        "PANELS:\n"
        "- [rgb] camera view\n"
        "- [depth] distance map (bright = near/wall, dark = far/open)\n"
        "- [confidence] reconstruction quality (bright = solid, dark = gaps/holes)\n"
        "- [map] top-down view rotating with you. Background: red = sparse gaps, "
        "green = good coverage. Blue dots = visited, green line = trail, "
        "red arrow = you (up = forward).\n"
        "\n"
        "CONTEXT:\n"
        "You see the full history of previous frames. Use them to remember what "
        "you were heading toward and why. If you spotted a gap and turned toward "
        "it, follow through.\n"
        "The [state] line shows your coverage %, step count, and wall distance.\n"
        "\n"
        "STRATEGY:\n"
        "1. FORWARD is your default. Move forward when the path is open.\n"
        "2. Seek out DARK regions in confidence and RED regions in the map — "
        "those are gaps that need you.\n"
        "3. After several forward steps, turn to scan for new gaps.\n"
        "4. When you see a gap in any direction, commit to reaching it.\n"
        "\n"
        "WALLS AND OBSTACLES:\n"
        "- If the depth panel is mostly bright or wall_distance in [state] is "
        "small, you are facing a wall.\n"
        "- A wall is NOT a reason to stop. Back up, turn, and go somewhere else.\n"
        "- If you see the same wall for several frames, you are stuck. Turn the "
        "OTHER direction or back up more aggressively.\n"
        "- Check [history]: if it shows repeating patterns like "
        "forward,back,forward,back you are oscillating. Break the loop by "
        "turning.\n"
        "\n"
        "STOPPING:\n"
        "- Check the coverage % in [state]. If coverage is below 90%, there are "
        "gaps you haven't reached yet. Do NOT stop.\n"
        "- Only stop when coverage is high AND you see no dark/red areas left.\n"
        "- Hitting a dead end is NEVER a reason to stop. Turn around.\n"
        "\n"
        'Reply ONLY with JSON: {"action": "<forward|back|turn_left|turn_right|move_up|move_down|stop>"}.'
    )

    def build(
        self,
        camera: Camera,
        coverage: CoverageState,
        trail: list[Pose],
        *,
        step: int = 0,
        budget: int = 0,
        action_history: list[str] | None = None,
        coverage_pct: float = 0.0,
        wall_distance: float | None = None,
    ) -> tuple[Observation, RenderResult]:
        """Render the view and assemble the observation.

        Returns the :class:`Observation` **and** the underlying
        :class:`RenderResult` (the explorer needs the render to judge
        degeneracy / record confidence).
        """
        result = self.renderer.render(camera)
        if self.depth_estimator is not None:
            da3_depth = self.depth_estimator.estimate(result.rgb)
            result = RenderResult(
                rgb=result.rgb,
                camera=result.camera,
                depth=da3_depth,
                alpha=result.alpha,
            )
        depth_panel = depth_to_uint8(result.depth)
        confidence_panel = self._confidence_panel(result.alpha)
        map_panel = self._body_frame_map(coverage, camera.pose, trail)
        state_line = self._state_line(camera.pose, step, budget, coverage_pct, wall_distance)

        parts = [self.prompt]
        if self.task:
            parts.append(f"TASK: {self.task}")
        if action_history:
            recent = action_history[-8:]
            parts.append(f"[history] last actions: {', '.join(recent)}")
        parts.append(state_line)

        obs = Observation(
            panels=[
                ("rgb", result.rgb),
                ("depth", depth_panel),
                ("confidence", confidence_panel),
                ("map", map_panel),
            ],
            prompt="\n".join(parts),
        )
        return obs, result

    @staticmethod
    def _confidence_panel(alpha: np.ndarray | None) -> np.ndarray:
        """Render alpha as an opacity map: bright = covered, dark = holes."""
        if alpha is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        h, w = alpha.shape
        gray = (alpha * 255).clip(0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def _state_line(
        self,
        pose: Pose,
        step: int,
        budget: int,
        coverage_pct: float = 0.0,
        wall_distance: float | None = None,
    ) -> str:
        step_str = f"{step}/{budget}" if budget > 0 else str(step)
        parts = [
            f"[state] step {step_str}",
            f"coverage {coverage_pct:.0%}",
            f"pose ({pose.position[0]:.2f},{pose.position[1]:.2f},{pose.position[2]:.2f})",
        ]
        if wall_distance is not None:
            parts.append(f"wall ahead {wall_distance:.2f}m")
        return "; ".join(parts)

    def _density_background(self, current: Pose, size: int, span: float) -> Image.Image:
        """Create the map background with density heatmap, rotated to body frame."""
        if not (hasattr(self.renderer, "cloud") and self.renderer.cloud.density_grid is not None):
            return Image.new("RGB", (size, size), (240, 240, 240))

        density_grid = self.renderer.cloud.density_grid
        db = self.renderer.cloud.density_bounds
        if db is None:
            return Image.new("RGB", (size, size), (240, 240, 240))

        lo, hi = db
        g = density_grid.shape[0]
        d = np.sqrt(density_grid)
        hm = np.zeros((g, g, 3), dtype=np.uint8)
        hm[:, :, 1] = (d * 200).clip(0, 255).astype(np.uint8)
        hm[:, :, 0] = (((1.0 - d) ** 2) * 160).clip(0, 255).astype(np.uint8)
        # density_grid axes are [x_bin, z_bin]; image needs [row=z, col=x]
        hm = np.transpose(hm, (1, 0, 2))[::-1].copy()

        # Scale the heatmap to world-space, then crop/rotate around the agent.
        cur_floor = floor_xy(current.position, self.up_axis)[0]
        fwd3 = current.heading(self.up_axis)
        fwd2 = floor_xy(fwd3, self.up_axis)[0]
        n2 = float(np.linalg.norm(fwd2))
        heading_deg = 0.0 if n2 < 1e-9 else float(np.degrees(np.arctan2(fwd2[0], fwd2[1])))

        world_w = hi[0] - lo[0]
        world_h = hi[2] - lo[2]
        if world_w < 1e-9 or world_h < 1e-9:
            return Image.new("RGB", (size, size), (240, 240, 240))

        big = Image.fromarray(hm).resize(
            (max(1, int(g * 4)), max(1, int(g * 4))), Image.Resampling.BILINEAR
        )
        bw, bh = big.size
        # Agent position in pixel coords on the big image
        cx = (cur_floor[0] - lo[0]) / world_w * bw
        cy = (1.0 - (cur_floor[1] - lo[2]) / world_h) * bh
        # Rotate around agent so heading points up
        rotated = big.rotate(heading_deg, center=(cx, cy), expand=True, fillcolor=(0, 0, 0))
        rw, rh = rotated.size
        new_cx = rw / 2 + (cx - bw / 2)
        new_cy = rh / 2 + (cy - bh / 2)
        # Crop a square around the agent matching the map span
        px_per_unit = bw / world_w
        half_px = int(span / 2.0 * px_per_unit)
        left = int(new_cx - half_px)
        top = int(new_cy - half_px)
        cropped = rotated.crop((left, top, left + 2 * half_px, top + 2 * half_px))
        return cropped.resize((size, size), Image.Resampling.BILINEAR)

    def _body_frame_map(
        self, coverage: CoverageState, current: Pose, trail: list[Pose]
    ) -> np.ndarray:
        size = self.map_size
        diag = float(np.linalg.norm(coverage.bounds_max - coverage.bounds_min))
        span = self.map_span if self.map_span is not None else max(diag, 1e-9)
        img = self._density_background(current, size, span)
        draw = ImageDraw.Draw(img)
        half_span = max(span / 2.0, 1e-9)
        px_per_unit = (size / 2.0) / half_span

        cur_floor = floor_xy(current.position, self.up_axis)[0]
        fwd3 = current.heading(self.up_axis)
        fwd2 = floor_xy(fwd3, self.up_axis)[0]
        n2 = float(np.linalg.norm(fwd2))
        fwd2 = np.array([1.0, 0.0]) if n2 < 1e-9 else fwd2 / n2
        right2 = np.array([fwd2[1], -fwd2[0]])

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
