"""The exploration loop (ADR-0003, ADR-0006).

:class:`Explorer` runs one local-control :term:`walk` from a seed pose, and
:meth:`Explorer.run_session` launches walks from many seeds into a shared
:class:`CoverageState`. The output is the union of walk trajectories, later
filtered (ADR-0008) into the deliverable.

An optional :class:`~gaussian_robot.events.EventSink` receives live
:class:`~gaussian_robot.events.SessionEvent`s for observability (e.g. the web
dashboard) without coupling the core to the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from gaussian_robot.events import (
    EventSink,
    SceneDescribeEvent,
    SessionEndEvent,
    SessionStartEvent,
    StepEvent,
    WalkEndEvent,
)
from gaussian_robot.metrics.coverage import (
    CoverageState,
    floor_coverage,
    floor_xy,
    pose_space_coverage,
)
from gaussian_robot.nav.action import Action, ActionSpace, apply_action, capped_forward_step
from gaussian_robot.nav.observation import ObservationBuilder, wall_distance_from_depth
from gaussian_robot.nav.robot import Robot
from gaussian_robot.nav.stop import (
    SessionContext,
    SessionStopPolicy,
    StopPolicy,
    WalkContext,
    session_stop_reason,
    walk_stop_reason,
)
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Pose
from gaussian_robot.splat.scene import SplatScene
from gaussian_robot.vlm.client import Decision, VLMClient
from gaussian_robot.vlm.observation import Observation

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedPose:
    """A pose a walk starts from, with its provenance.

    ``kind`` records where the seed came from so the deliverable and the UI can
    distinguish a real captured viewpoint from a synthesised fallback:

    - ``"capture"`` — a real camera the splat was reconstructed from (best).
    - ``"density"`` — sampled from the reconstructed-density grid (a guess).
    - ``"grid"`` — a plain floor-grid position (last-resort guess).
    - ``"origin_fallback"`` — synthesised at the validated origin, facing out.
    """

    pose: Pose
    kind: str = "capture"


@dataclass
class WalkStep:
    """One recorded step of a walk."""

    pose: Pose
    action: Action
    novelty: float
    degenerate: bool
    raw_text: str = ""
    blocked: bool = False


@dataclass
class WalkResult:
    """The trajectory produced by one walk."""

    walk_id: str
    steps: list[WalkStep] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def poses(self) -> list[Pose]:
        return [s.pose for s in self.steps]


def _render_degenerate(result: RenderResult, *, min_finite_frac: float = 0.25) -> bool:
    """Heuristic: a render is degenerate if depth is mostly non-finite."""
    if result.depth is None:
        return False
    finite_frac: float = float(np.isfinite(result.depth).mean())
    return finite_frac < min_finite_frac


def _floor_array(poses: list[Pose], up_axis: str) -> np.ndarray:
    if not poses:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([floor_xy(p.position, up_axis)[0] for p in poses], dtype=np.float64)


@dataclass
class Explorer:
    """Runs walks and sessions.

    The renderer, VLM and observation builder are injected (seam per ADR-0001).
    Walk-level ``walk_policies`` are OR-composed and queried each step; the
    ``max_steps`` cap is an additional safety net. Session-level
    ``session_policies`` are evaluated between walks. Set ``event_sink`` to
    receive live events.
    """

    scene: SplatScene
    renderer: Renderer
    vlm: VLMClient
    observation_builder: ObservationBuilder
    action_space: ActionSpace
    coverage_radius: float
    walk_policies: list[StopPolicy] = field(default_factory=list)
    session_policies: list[SessionStopPolicy] = field(default_factory=list)
    max_steps: int = 40
    event_sink: EventSink | None = None

    def _describe_step(
        self,
        robot: Robot,
        result: WalkResult,
        action_history: list[str],
        *,
        walk_id: str,
        step: int,
        raw_text: str = "",
    ) -> str:
        """Render the view, describe the scene, and record the describe step."""
        desc_obs, _ = self.observation_builder.build_describe(robot.camera())
        description = self.vlm.describe(desc_obs)
        _log.info("Scene description (walk=%s step=%d): %s", walk_id, step, description)
        if self.event_sink is not None:
            self.event_sink(SceneDescribeEvent(walk_id=walk_id, step=step, description=description))
        action_history.append(Action.DESCRIBE.value)
        result.steps.append(
            WalkStep(
                pose=robot.pose,
                action=Action.DESCRIBE,
                novelty=0.0,
                degenerate=False,
                raw_text=raw_text or description,
            )
        )
        return description

    def run_walk(
        self, seed_pose: Pose, coverage: CoverageState, *, walk_id: str = ""
    ) -> WalkResult:
        self.vlm.reset()
        robot = Robot(scene=self.scene, pose=seed_pose)
        for p in self.walk_policies:
            p.reset()

        scene_description = ""

        trail: list[Pose] = [seed_pose]
        novelty_seed = coverage.novelty(seed_pose)
        coverage.add_pose(seed_pose, walk_id=walk_id)
        result = WalkResult(walk_id=walk_id)
        result.steps.append(
            WalkStep(pose=seed_pose, action=Action.STOP, novelty=novelty_seed, degenerate=False)
        )

        action_history: list[str] = []
        stop_reason = "step_budget"
        for step_idx in range(self.max_steps):
            if step_idx == 0:
                scene_description = self._describe_step(
                    robot, result, action_history, walk_id=walk_id, step=1
                )
                continue

            camera = robot.camera()
            cov = floor_coverage(coverage, radius=self.coverage_radius)
            observation, render = self.observation_builder.build(
                camera,
                coverage,
                trail,
                step=step_idx + 1,
                budget=self.max_steps,
                action_history=action_history,
                coverage_pct=cov,
                scene_description=scene_description,
            )
            decision = self.vlm.act(observation)
            action = decision.action

            if action is Action.DESCRIBE:
                scene_description = self._describe_step(
                    robot,
                    result,
                    action_history,
                    walk_id=walk_id,
                    step=step_idx + 1,
                    raw_text=decision.raw_text,
                )
                continue

            action_history.append(action.value)
            # Free distance ahead from the *current* metric render caps a forward
            # step so the camera halts short of obstacles instead of burrowing in.
            clearance = wall_distance_from_depth(render.depth)
            next_pose = apply_action(
                robot.pose, action, self.action_space, self.scene.up_axis, clearance=clearance
            )
            blocked = self._forward_blocked(action, clearance)
            novelty_next = coverage.novelty(next_pose)
            degenerate = self._is_degenerate(next_pose, render)

            ctx = WalkContext(
                step=step_idx + 1,
                action=action,
                novelty=novelty_next,
                pose=next_pose,
                degenerate=degenerate,
            )
            for p in self.walk_policies:
                p.update(ctx)

            result.steps.append(
                WalkStep(
                    pose=next_pose,
                    action=action,
                    novelty=novelty_next,
                    degenerate=degenerate,
                    raw_text=decision.raw_text,
                    blocked=blocked,
                )
            )

            if action is not Action.STOP and not degenerate and not blocked:
                robot.move(next_pose)
                conf = float(render.alpha.mean()) if render.alpha is not None else 1.0
                coverage.add_pose(next_pose, walk_id=walk_id, confidence=conf)
                trail.append(next_pose)

            self._emit_step(
                walk_id,
                step_idx + 1,
                observation,
                decision,
                action,
                next_pose,
                novelty_next,
                degenerate,
                coverage,
                trail,
                blocked=blocked,
            )

            reason = walk_stop_reason(self.walk_policies)
            if reason is not None:
                stop_reason = reason
                break

        result.stop_reason = stop_reason
        if self.event_sink is not None:
            self.event_sink(
                WalkEndEvent(walk_id=walk_id, reason=stop_reason, steps=len(result.steps))
            )
        return result

    def _forward_blocked(self, action: Action, clearance: float | None) -> bool:
        """True when a FORWARD step is capped to (near) zero by an obstacle ahead."""
        if action is not Action.FORWARD or clearance is None:
            return False
        return capped_forward_step(self.action_space.step, clearance) <= 1e-6

    def run_session(
        self,
        seeds: list[SeedPose],
        coverage: CoverageState,
        *,
        requested_seeds: int | None = None,
    ) -> list[WalkResult]:
        """Launch a walk per seed into the shared ``coverage`` until session stop.

        Each :class:`SeedPose` carries its provenance (``kind``); walks are
        identified by ``walk{i}`` (a walk id, not a seed). Session policies are
        evaluated **after** each walk against the coverage that walk produced —
        so the gain attributed to walk *i* is genuinely walk *i*'s, the final
        walk is also subject to the policies, and the recorded ``reason`` names
        the policy that fired (or ``seeds_exhausted`` when the seeds run out).
        ``requested_seeds`` is how many were asked for (defaults to the number
        actually launched) so the UI can report rejections.
        """
        if self.event_sink is not None:
            self.event_sink(
                SessionStartEvent(
                    bounds_min=coverage.bounds_min,
                    bounds_max=coverage.bounds_max,
                    up_axis=coverage.up_axis,
                    total_seeds=len(seeds),
                    seed_floor=_floor_array([s.pose for s in seeds], coverage.up_axis),
                    seed_kinds=[s.kind for s in seeds],
                    requested_seeds=len(seeds) if requested_seeds is None else requested_seeds,
                )
            )

        results: list[WalkResult] = []
        prev_cov = floor_coverage(coverage, radius=self.coverage_radius)
        reason = "seeds_exhausted"
        for i, seed in enumerate(seeds):
            results.append(self.run_walk(seed.pose, coverage, walk_id=f"walk{i}"))
            cur_cov = floor_coverage(coverage, radius=self.coverage_radius)
            gain = cur_cov - prev_cov
            prev_cov = cur_cov
            ctx = SessionContext(
                state=coverage,
                walks_completed=i + 1,
                total_seeds=len(seeds),
                last_batch_coverage_gain=gain,
            )
            fired = session_stop_reason(self.session_policies, ctx)
            if fired is not None:
                reason = fired
                break

        if self.event_sink is not None:
            total_steps = sum(len(r.steps) for r in results)
            self.event_sink(
                SessionEndEvent(
                    reason=reason,
                    total_steps=total_steps,
                    total_poses=len(coverage),
                )
            )
        return results

    def _emit_step(
        self,
        walk_id: str,
        step: int,
        observation: Observation,
        decision: Decision,
        action: Action,
        pose: Pose,
        novelty: float,
        degenerate: bool,
        coverage: CoverageState,
        trail: list[Pose],
        *,
        blocked: bool = False,
    ) -> None:
        if self.event_sink is None:
            return
        cov_floor = floor_coverage(coverage, radius=self.coverage_radius)
        cov_ps = pose_space_coverage(coverage, radius=self.coverage_radius)
        self.event_sink(
            StepEvent(
                walk_id=walk_id,
                step=step,
                budget=self.max_steps,
                observation=observation,
                decision=decision,
                action=action,
                pose=pose,
                novelty=novelty,
                degenerate=degenerate,
                coverage_floor=cov_floor,
                coverage_pose_space=cov_ps,
                sampled_floor=coverage.floor_positions(),
                trail_floor=_floor_array(trail, self.scene.up_axis),
                blocked=blocked,
            )
        )

    def _is_degenerate(self, pose: Pose, render: RenderResult) -> bool:
        out_of_bounds = bool(
            np.any(pose.position < self.scene.bounds.min)
            or np.any(pose.position > self.scene.bounds.max)
        )
        return out_of_bounds or _render_degenerate(render)
