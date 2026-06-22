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
)
from gaussian_robot.metrics.coverage import (
    CoverageState,
    floor_coverage,
    floor_xy,
    pose_space_coverage,
)
from gaussian_robot.nav.action import Action, ActionSpace, apply_action
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.nav.robot import Robot
from gaussian_robot.nav.stop import (
    SessionContext,
    SessionStopPolicy,
    StopPolicy,
    WalkContext,
    any_session_stop,
    any_walk_stop,
)
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Pose
from gaussian_robot.splat.scene import SplatScene
from gaussian_robot.vlm.client import Decision, VLMClient
from gaussian_robot.vlm.observation import Observation

_log = logging.getLogger(__name__)


@dataclass
class WalkStep:
    """One recorded step of a walk."""

    pose: Pose
    action: Action
    novelty: float
    degenerate: bool
    raw_text: str = ""


@dataclass
class WalkResult:
    """The trajectory produced by one walk."""

    seed_id: str
    steps: list[WalkStep] = field(default_factory=list)

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
        seed_id: str,
        step: int,
        raw_text: str = "",
    ) -> str:
        """Render the view, describe the scene, and record the describe step."""
        desc_obs, _ = self.observation_builder.build_describe(robot.camera())
        description = self.vlm.describe(desc_obs)
        _log.info("Scene description (seed=%s step=%d): %s", seed_id, step, description)
        if self.event_sink is not None:
            self.event_sink(SceneDescribeEvent(seed_id=seed_id, step=step, description=description))
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
        self, seed_pose: Pose, coverage: CoverageState, *, seed_id: str = ""
    ) -> WalkResult:
        self.vlm.reset()
        robot = Robot(scene=self.scene, pose=seed_pose)
        for p in self.walk_policies:
            p.reset()

        scene_description = ""

        trail: list[Pose] = [seed_pose]
        novelty_seed = coverage.novelty(seed_pose)
        coverage.add_pose(seed_pose, seed_id=seed_id)
        result = WalkResult(seed_id=seed_id)
        result.steps.append(
            WalkStep(pose=seed_pose, action=Action.STOP, novelty=novelty_seed, degenerate=False)
        )

        action_history: list[str] = []
        prev_render: RenderResult | None = None
        for step_idx in range(self.max_steps):
            if step_idx == 0:
                scene_description = self._describe_step(
                    robot, result, action_history, seed_id=seed_id, step=1
                )
                continue

            camera = robot.camera()
            cov = floor_coverage(coverage, radius=self.coverage_radius)
            wall_dist = self._wall_distance(prev_render)
            observation, render = self.observation_builder.build(
                camera,
                coverage,
                trail,
                step=step_idx + 1,
                budget=self.max_steps,
                action_history=action_history,
                coverage_pct=cov,
                wall_distance=wall_dist,
                scene_description=scene_description,
            )
            decision = self.vlm.act(observation)
            action = decision.action

            if action is Action.DESCRIBE:
                scene_description = self._describe_step(
                    robot,
                    result,
                    action_history,
                    seed_id=seed_id,
                    step=step_idx + 1,
                    raw_text=decision.raw_text,
                )
                prev_render = render
                continue

            action_history.append(action.value)
            next_pose = apply_action(robot.pose, action, self.action_space, self.scene.up_axis)
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
                )
            )

            if action is not Action.STOP and not degenerate:
                robot.move(next_pose)
                coverage.add_pose(next_pose, seed_id=seed_id)
                trail.append(next_pose)

            self._emit_step(
                seed_id,
                step_idx + 1,
                observation,
                decision,
                action,
                next_pose,
                novelty_next,
                degenerate,
                coverage,
                trail,
            )

            prev_render = render

            if any_walk_stop(self.walk_policies):
                break

        return result

    def run_session(self, seed_poses: list[Pose], coverage: CoverageState) -> list[WalkResult]:
        """Launch a walk per seed into the shared ``coverage`` until session stop."""
        if self.event_sink is not None:
            self.event_sink(
                SessionStartEvent(
                    bounds_min=coverage.bounds_min,
                    bounds_max=coverage.bounds_max,
                    up_axis=coverage.up_axis,
                    total_seeds=len(seed_poses),
                )
            )

        results: list[WalkResult] = []
        prev_cov = floor_coverage(coverage, radius=self.coverage_radius)
        stopped = False
        for i, seed in enumerate(seed_poses):
            gain = 0.0
            if i > 0:
                cur_cov = floor_coverage(coverage, radius=self.coverage_radius)
                gain = cur_cov - prev_cov
                prev_cov = cur_cov
            ctx = SessionContext(
                state=coverage,
                walks_completed=i,
                total_seeds=len(seed_poses),
                last_batch_coverage_gain=gain,
            )
            if i > 0 and any_session_stop(self.session_policies, ctx):
                stopped = True
                break
            results.append(self.run_walk(seed, coverage, seed_id=f"seed{i}"))

        if self.event_sink is not None:
            total_steps = sum(len(r.steps) for r in results)
            self.event_sink(
                SessionEndEvent(
                    reason="session_policy" if stopped else "completed",
                    total_steps=total_steps,
                    total_poses=len(coverage),
                )
            )
        return results

    def _emit_step(
        self,
        seed_id: str,
        step: int,
        observation: Observation,
        decision: Decision,
        action: Action,
        pose: Pose,
        novelty: float,
        degenerate: bool,
        coverage: CoverageState,
        trail: list[Pose],
    ) -> None:
        if self.event_sink is None:
            return
        cov_floor = floor_coverage(coverage, radius=self.coverage_radius)
        cov_ps = pose_space_coverage(coverage, radius=self.coverage_radius)
        self.event_sink(
            StepEvent(
                seed_id=seed_id,
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
            )
        )

    def _wall_distance(self, render: RenderResult | None) -> float | None:
        """Median depth in the horizontal band ahead, or None if unavailable."""
        if render is None or render.depth is None:
            return None
        h, w = render.depth.shape
        band = render.depth[2 * h // 5 : 3 * h // 5, :]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def _is_degenerate(self, pose: Pose, render: RenderResult) -> bool:
        out_of_bounds = bool(
            np.any(pose.position < self.scene.bounds.min)
            or np.any(pose.position > self.scene.bounds.max)
        )
        return out_of_bounds or _render_degenerate(render)
