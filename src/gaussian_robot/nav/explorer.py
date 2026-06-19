"""The exploration loop (ADR-0003, ADR-0006).

:class:`Explorer` runs one local-control :term:`walk` from a seed pose, and
:meth:`Explorer.run_session` launches walks from many seeds into a shared
:class:`CoverageState`. The output is the union of walk trajectories, later
filtered (ADR-0008) into the deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaussian_robot.metrics.coverage import CoverageState, floor_coverage
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
from gaussian_robot.vlm.client import VLMClient


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


@dataclass
class Explorer:
    """Runs walks and sessions.

    The renderer, VLM and observation builder are injected (seam per ADR-0001).
    Walk-level ``walk_policies`` are OR-composed and queried each step; the
    ``max_steps`` cap is an additional safety net. Session-level
    ``session_policies`` are evaluated between walks.
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

    def run_walk(
        self, seed_pose: Pose, coverage: CoverageState, *, seed_id: str = ""
    ) -> WalkResult:
        robot = Robot(scene=self.scene, pose=seed_pose)
        for p in self.walk_policies:
            p.reset()

        trail: list[Pose] = [seed_pose]
        novelty_seed = coverage.novelty(seed_pose)
        coverage.add_pose(seed_pose, seed_id=seed_id)
        result = WalkResult(seed_id=seed_id)
        result.steps.append(
            WalkStep(pose=seed_pose, action=Action.STOP, novelty=novelty_seed, degenerate=False)
        )

        for step_idx in range(self.max_steps):
            camera = robot.camera()
            observation, render = self.observation_builder.build(
                camera, coverage, trail, step=step_idx + 1, budget=self.max_steps
            )
            decision = self.vlm.act(observation)
            action = decision.action

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

            if action is not Action.STOP:
                robot.move(next_pose)
                coverage.add_pose(next_pose, seed_id=seed_id)
                trail.append(next_pose)

            if any_walk_stop(self.walk_policies):
                break

        return result

    def run_session(self, seed_poses: list[Pose], coverage: CoverageState) -> list[WalkResult]:
        """Launch a walk per seed into the shared ``coverage`` until session stop."""
        results: list[WalkResult] = []
        prev_cov = floor_coverage(coverage, radius=self.coverage_radius)
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
                break
            results.append(self.run_walk(seed, coverage, seed_id=f"seed{i}"))
        return results

    def _is_degenerate(self, pose: Pose, render: RenderResult) -> bool:
        out_of_bounds = bool(
            np.any(pose.position < self.scene.bounds.min)
            or np.any(pose.position > self.scene.bounds.max)
        )
        return out_of_bounds or _render_degenerate(render)
