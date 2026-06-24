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


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stack a ``(H, W)`` uint8 plane into an ``(H, W, 3)`` RGB image."""
    return np.stack([gray, gray, gray], axis=-1)


def wall_distance_from_depth(depth: np.ndarray | None) -> float | None:
    """Median depth in the central horizontal band — the free distance ahead.

    Operates on **metric** render depth (the renderer's z-buffer), never a
    monocular estimate, so the result is in scene units and safe to use for
    step-capping and the ``wall ahead`` state line. Returns ``None`` when no
    finite depth is available.
    """
    if depth is None:
        return None
    h, _ = depth.shape
    band = depth[2 * h // 5 : 3 * h // 5, :]
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


_ROTATION_ACTIONS = frozenset({"turn_left", "turn_right", "look_up", "look_down"})


def _rotation_streak(action_history: list[str] | None) -> int:
    """How many trailing actions were pure rotations (no translation) — anti-spin cue."""
    if not action_history:
        return 0
    streak = 0
    for a in reversed(action_history):
        if a in _ROTATION_ACTIONS:
            streak += 1
        else:
            break
    return streak


def frontier_mask(
    density_grid: np.ndarray, *, observed_min: float = 0.10, under_frac: float = 0.5
) -> np.ndarray:
    """Boolean grid of **reconstruction frontiers**: real holes worth new views.

    A frontier is an **under-sampled** cell — density below ``under_frac`` of the
    *typical observed* density (the median over reconstructed cells) — that is
    4-adjacent to a reconstructed cell (``> observed_min``). This captures both
    the reconstruction's outer edge (empty cells next to observed ones) **and**
    interior under-sampled pockets (sparse-but-not-empty regions like smeared
    ground/foliage), so a gap *near the agent* is found — not only the far scene
    boundary. Fully-reconstructed cells and open void with no observed neighbour
    never qualify.
    """
    observed = density_grid > observed_min
    if not observed.any():
        return np.zeros(density_grid.shape, dtype=bool)
    under_max = under_frac * float(np.median(density_grid[observed]))
    adj = np.zeros(density_grid.shape, dtype=bool)
    adj[:-1, :] |= observed[1:, :]
    adj[1:, :] |= observed[:-1, :]
    adj[:, :-1] |= observed[:, 1:]
    adj[:, 1:] |= observed[:, :-1]
    return (density_grid < under_max) & adj


def _line_of_sight_clear(
    a: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    occ_threshold: float,
    *,
    samples: int = 24,
) -> bool:
    """True if the floor segment ``a -> b`` (world x,z) avoids occupied cells.

    Heuristic reachability: a cell is "occupied" (a wall/tree/solid you can't cross)
    when its top-down density is ``>= occ_threshold``. Samples the segment interior
    (endpoints skipped — the gap is sparse by definition, the agent sits in free
    space). A coarse proxy on a top-down grid, used only to *prefer* a reachable gap.
    """
    lo, hi = bounds
    wx, wz = hi[0] - lo[0], hi[2] - lo[2]
    g = grid.shape[0]
    if wx <= 0 or wz <= 0:
        return True
    for t in np.linspace(0.1, 0.9, samples):
        p = a * (1 - t) + b * t
        ix = int((p[0] - lo[0]) / wx * g)
        iz = int((p[1] - lo[2]) / wz * g)
        if 0 <= ix < g and 0 <= iz < g and grid[ix, iz] >= occ_threshold:
            return False
    return True


def frontier_floor_positions(renderer: Renderer) -> np.ndarray:
    """World floor ``(K, 2)`` centres of every reconstruction frontier cell.

    Computed from the renderer's (static) density grid; empty ``(0, 2)`` when the
    renderer has no density grid or the scene has no frontiers.
    """
    cloud = getattr(renderer, "cloud", None)
    grid = getattr(cloud, "density_grid", None)
    db = getattr(cloud, "density_bounds", None)
    if grid is None or db is None:
        return np.empty((0, 2), dtype=np.float64)
    mask = frontier_mask(grid)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float64)
    lo, hi = db
    g = grid.shape[0]
    ix, iz = np.nonzero(mask)  # grid axes [x_bin, z_bin]
    gx = lo[0] + (ix + 0.5) / g * (hi[0] - lo[0])
    gz = lo[2] + (iz + 0.5) / g * (hi[2] - lo[2])
    return np.stack([gx, gz], axis=1).astype(np.float64)


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
    return _gray_to_rgb(gray)


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
    describe_prompt: str = (
        "This is a render of a 3D Gaussian Splatting (3DGS) reconstruction, not a real photo. "
        "Blurriness, smearing, or ghosting are reconstruction artifacts — areas where the 3DGS "
        "model lacks enough training views and needs improvement.\n\n"
        "Describe the scene in 3-4 sentences:\n"
        "1. What kind of space or environment is this (room, corridor, outdoor area, etc.)?\n"
        "2. What are the main objects, surfaces, and structures visible?\n"
        "3. Which areas look sharp and well-reconstructed vs. blurry/artifact-ridden?\n"
        "4. Based on what you can see, which directions or areas should be prioritised "
        "for new viewpoints to improve the reconstruction?"
    )
    prompt: str = (
        "You are a robot inside a 3D Gaussian-Splat reconstruction. Your job is to find "
        "under-reconstructed spots and MARK them: each MARK records your current viewpoint as a "
        "proposed new camera view to fill the scene in. The marked poses ARE your deliverable — "
        "you are collecting them toward a target shown in [state] as 'marks N/target'. Travel to "
        "gaps (under-sampled regions you can reach), and whenever you arrive facing a blurry / "
        "under-observed region, MARK before moving on. Keep moving and keep marking until you hit "
        "the target; do NOT just wander or stare at things without marking.\n"
        "\n"
        "WHAT A GAP IS (and is NOT):\n"
        "- A gap is OPEN, reachable space that is under-observed — you can travel toward it.\n"
        "- A wall, tree, object or any surface right in front of you is NOT a gap. It only looks "
        "blurry because you are too close. It is an OBSTACLE: go around it, do not push into it.\n"
        "- Being stuck/blocked against something is never a goal. If you can't move forward, the "
        "answer is to turn or back away toward open space — never to sit there.\n"
        "\n"
        "PANELS:\n"
        "- [rgb] camera view.\n"
        "- [depth] distance ahead (bright = near/surface, dark = far/open). Bright everywhere = a "
        "surface right in front of you.\n"
        "- [confidence] render alpha: bright = solid geometry, dark = holes/missing surface.\n"
        "- [map] LOCAL top-down view, centred on you, rotating with you (up = forward). "
        "Background = training density: red = sparse, green = dense. Blue dots = visited, "
        "amber line = your trail, red arrow = you, MAGENTA diamond = the nearest GAP to head "
        "for (clamped to the edge as a pointer when it's beyond the view). Grey rings mark "
        "distance; the 'N' tick is fixed world-north.\n"
        "\n"
        "NOTE: the [depth] panel may use relative (non-metric) scale if a monocular estimator is "
        "active. Use wall_distance in [state] for metric distance, and 'nearest gap' in [state] "
        "for the bearing/distance to the closest gap.\n"
        "\n"
        "STRATEGY:\n"
        "1. FORWARD IS YOUR DEFAULT. Only turning never gets you anywhere — turns and looks do "
        "NOT change your position, only forward/back do. If the way ahead is open (wall_distance "
        "is not small), move FORWARD to make progress and reveal new area.\n"
        "2. NEVER turn more than twice in a row. After at most two turns you MUST move forward or "
        "back. Do not oscillate turn_left/turn_right — pick a direction and commit by moving.\n"
        "3. Gap is a soft bias, not a precondition: if the 'nearest gap' bearing is well off to "
        "one side, turn toward it once or twice and then advance. You do NOT need it perfectly "
        "centred — once it's roughly ahead, MOVE.\n"
        "4. Obstacle handling: if wall_distance is small or [depth] is mostly bright you face a "
        "surface — turn once or twice OR back up to find an open direction, then move. Don't keep "
        "turning in place.\n"
        "5. MARK often — it is your main output. Whenever you reach an under-reconstructed spot "
        "(dark [confidence] holes, smeared ground/foliage, a gap you traveled to) and it fills "
        "your view, emit MARK to record this viewpoint as a proposed new view. One mark per "
        "distinct spot, then move on to the next. Watch 'marks N/target' in [state] and keep "
        "marking until you reach the target.\n"
        "6. LOOK_UP / LOOK_DOWN to check ceilings/floors when entering a new area; DESCRIBE if "
        "disoriented.\n"
        "\n"
        "STOPPING:\n"
        "- Keep going while there is open space to explore or a gap is reported in [state].\n"
        "- Only stop when coverage is high and there is nowhere new to go. A wall, a dead end, or "
        "not seeing a gap is NEVER a reason to stop — move forward or turn around and keep going.\n"
        "\n"
        'Reply ONLY with JSON: {"action": "<forward|back|turn_left|turn_right|move_up|move_down|look_up|look_down|mark|describe|stop>"}.'
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
        scene_description: str = "",
        marks: int = 0,
        mark_target: int = 0,
    ) -> tuple[Observation, RenderResult]:
        """Render the view and assemble the observation.

        Returns the :class:`Observation` **and** the underlying
        :class:`RenderResult` (the explorer needs the render to judge
        degeneracy / record confidence). The returned result keeps the
        renderer's **metric** depth even when a monocular estimator is active —
        the estimate is used only for the human-facing depth *panel*, never for
        geometry decisions (degeneracy, wall distance, step-capping).
        """
        result = self.renderer.render(camera)
        display_depth = result.depth
        if self.depth_estimator is not None:
            display_depth = self.depth_estimator.estimate(result.rgb)
        depth_panel = depth_to_uint8(display_depth)
        confidence_panel = self._confidence_panel(result.alpha)
        cur_floor = floor_xy(camera.pose.position, self.up_axis)[0]
        gap_xy = self._nearest_gap(cur_floor)
        gap_info = self._gap_bearing(camera.pose, cur_floor, gap_xy)
        map_panel = self._body_frame_map(coverage, camera.pose, trail, gap_xy=gap_xy)
        wall_distance = wall_distance_from_depth(result.depth)
        state_line = self._state_line(
            camera.pose, step, budget, coverage_pct, wall_distance, gap_info, marks, mark_target
        )

        parts = [self.prompt]
        if scene_description:
            parts.append(f"SCENE DESCRIPTION: {scene_description}")
        if self.task:
            parts.append(f"TASK: {self.task}")
        if action_history:
            recent = action_history[-8:]
            parts.append(f"[history] last actions: {', '.join(recent)}")
        parts.append(state_line)
        spin = _rotation_streak(action_history)
        if spin >= 2:
            parts.append(
                f"[!] You have turned {spin}× in a row WITHOUT moving — turning does not change "
                "your position. Move FORWARD now (or BACK if blocked); do not turn again."
            )

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

    def build_describe(self, camera: Camera) -> tuple[Observation, RenderResult]:
        """Render the view and build an observation for scene description only.

        Returns a lightweight observation with just the RGB panel and the
        describe prompt, plus the underlying :class:`RenderResult`.
        """
        result = self.renderer.render(camera)
        obs = Observation(
            panels=[("rgb", result.rgb)],
            prompt=self.describe_prompt,
        )
        return obs, result

    @staticmethod
    def _confidence_panel(alpha: np.ndarray | None) -> np.ndarray:
        """Render alpha as an opacity map: bright = covered, dark = holes."""
        if alpha is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        gray = (alpha * 255).clip(0, 255).astype(np.uint8)
        return _gray_to_rgb(gray)

    def _state_line(
        self,
        pose: Pose,
        step: int,
        budget: int,
        coverage_pct: float = 0.0,
        wall_distance: float | None = None,
        gap_info: tuple[float, float] | None = None,
        marks: int = 0,
        mark_target: int = 0,
    ) -> str:
        step_str = f"{step}/{budget}" if budget > 0 else str(step)
        parts = [
            f"[state] step {step_str}",
            f"coverage {coverage_pct:.0%}",
            f"pose ({pose.position[0]:.2f},{pose.position[1]:.2f},{pose.position[2]:.2f})",
        ]
        if mark_target > 0:
            parts.append(f"marks {marks}/{mark_target}")
        if wall_distance is not None:
            parts.append(f"wall ahead {wall_distance:.2f}m")
        if gap_info is not None:
            dist, bearing = gap_info
            if abs(bearing) < 12:
                parts.append(f"nearest gap {dist:.1f}m ahead")
            else:
                side = "right" if bearing > 0 else "left"
                parts.append(f"nearest gap {abs(bearing):.0f}° {side}, {dist:.1f}m")
        else:
            parts.append("nearest gap: none in range")
        return "; ".join(parts)

    def _nearest_gap(self, cur_floor: np.ndarray) -> np.ndarray | None:
        """World floor ``(x, z)`` of the best nearby reconstruction frontier, or None.

        Prefers the nearest frontier reachable by a clear straight path (no occupied
        cells crossed); falls back to the plain nearest frontier when none is clear.
        """
        pts = frontier_floor_positions(self.renderer)
        if pts.shape[0] == 0:
            return None
        order = np.argsort(((pts - cur_floor) ** 2).sum(axis=1))
        cloud = getattr(self.renderer, "cloud", None)
        grid = getattr(cloud, "density_grid", None)
        db = getattr(cloud, "density_bounds", None)
        if grid is not None and db is not None:
            dense = grid[grid > 0.1]
            if dense.size:
                occ = float(np.quantile(dense, 0.85))  # top ~15% density = likely solid
                for i in order[:24]:  # check only the nearest handful
                    if _line_of_sight_clear(cur_floor, pts[i], grid, db, occ):
                        return np.asarray(pts[i], dtype=np.float64)
        return np.asarray(pts[int(order[0])], dtype=np.float64)

    def nearest_gap(self, pose: Pose) -> tuple[float, float] | None:
        """(distance, signed bearing°) to the nearest frontier from ``pose``, or None."""
        cur = floor_xy(pose.position, self.up_axis)[0]
        return self._gap_bearing(pose, cur, self._nearest_gap(cur))

    def _gap_bearing(
        self, pose: Pose, cur_floor: np.ndarray, gap_xy: np.ndarray | None
    ) -> tuple[float, float] | None:
        """(distance, signed bearing°) to ``gap_xy`` in the agent frame (+ = right)."""
        if gap_xy is None:
            return None
        fwd2 = floor_xy(pose.heading(self.up_axis), self.up_axis)[0]
        n2 = float(np.linalg.norm(fwd2))
        fwd2 = np.array([1.0, 0.0]) if n2 < 1e-9 else fwd2 / n2
        right2 = np.array([fwd2[1], -fwd2[0]])
        v = gap_xy - cur_floor
        ahead = float(v @ fwd2)
        right = float(v @ right2)
        return (float(np.hypot(ahead, right)), float(np.degrees(np.arctan2(right, ahead))))

    def _density_background(self, current: Pose, size: int, span: float) -> Image.Image:
        """Create the map background with density heatmap, rotated to body frame."""

        def flat() -> Image.Image:
            return Image.new("RGB", (size, size), (240, 240, 240))

        if not (hasattr(self.renderer, "cloud") and self.renderer.cloud.density_grid is not None):
            return flat()

        density_grid = self.renderer.cloud.density_grid
        db = self.renderer.cloud.density_bounds
        if db is None:
            return flat()

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
            return flat()

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

    def _map_span(self, coverage: CoverageState) -> float:
        """World units spanned across the map.

        Uses the explicit ``map_span`` when set (an agent-centric local window);
        otherwise falls back to the scene's **floor-plane** diagonal (the up axis
        is excluded so a tall room doesn't shrink the footprint).
        """
        if self.map_span is not None:
            return max(self.map_span, 1e-9)
        extent = np.abs(coverage.bounds_max - coverage.bounds_min)
        floor_extent = floor_xy(extent, self.up_axis)[0]
        return max(float(np.linalg.norm(floor_extent)), 1e-9)

    def _body_frame_map(
        self,
        coverage: CoverageState,
        current: Pose,
        trail: list[Pose],
        *,
        gap_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        size = self.map_size
        span = self._map_span(coverage)
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

        cx = cy = size / 2.0
        self._draw_scale_rings(draw, cx, cy, half_span, px_per_unit)

        sampled = coverage.floor_positions()
        r_dot = max(3, size // 128)
        for s in sampled:
            x, y = to_pixel(s)
            if -size <= x <= 2 * size and -size <= y <= 2 * size:
                draw.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot), fill=(40, 90, 200))

        if len(trail) >= 2:
            line = [to_pixel(floor_xy(p.position, self.up_axis)[0]) for p in trail]
            draw.line(line, fill=(245, 190, 40), width=max(3, size // 170))

        # Nearest reconstruction gap: a magenta diamond, clamped to the panel edge
        # when it lies outside the view so it always points the way to the hole.
        if gap_xy is not None:
            gx, gy = to_pixel(gap_xy)
            m = max(6, size // 64)
            gx = float(np.clip(gx, m, size - m))
            gy = float(np.clip(gy, m, size - m))
            draw.polygon(
                ((gx, gy - m), (gx + m, gy), (gx, gy + m), (gx - m, gy)), fill=(230, 70, 230)
            )

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

        self._draw_north_tick(draw, cx, cy, fwd2, right2, size)
        draw.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180))
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _draw_scale_rings(
        draw: ImageDraw.ImageDraw, cx: float, cy: float, half_span: float, px_per_unit: float
    ) -> None:
        """Three faint concentric rings as a distance reference; label the outer radius."""
        for frac in (1 / 3, 2 / 3, 1.0):
            r = half_span * frac * px_per_unit
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(150, 150, 150))
        label = f"{half_span:.1f}m" if half_span < 10 else f"{half_span:.0f}m"
        draw.text((cx + 3, cy - half_span * px_per_unit + 2), label, fill=(170, 170, 170))

    def _draw_north_tick(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        fwd2: np.ndarray,
        right2: np.ndarray,
        size: int,
    ) -> None:
        """A small fixed world-north marker on the body-frame map.

        World-north is +(first floor axis). Projected into the agent frame it
        swings as the agent turns (rotation cue) but holds when the agent only
        translates (so turns and steps look different on the rotating map).
        """
        north2 = np.array([1.0, 0.0])  # +(first floor axis) in floor coords
        ahead = float(north2 @ fwd2)
        right = float(north2 @ right2)
        radius = size * 0.44
        nx = cx + right * radius
        ny = cy - ahead * radius
        draw.line(
            (cx + right * radius * 0.88, cy - ahead * radius * 0.88, nx, ny),
            fill=(120, 160, 230),
            width=max(2, size // 200),
        )
        draw.text((nx - 3, ny - 6), "N", fill=(120, 160, 230))
