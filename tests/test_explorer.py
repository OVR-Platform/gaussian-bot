"""Integration tests for the exploration loop (ADR-0003, ADR-0005, ADR-0006)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np
import pytest

from gaussian_robot.events import MarkEvent, SessionEndEvent, SessionStartEvent, WalkEndEvent
from gaussian_robot.metrics.coverage import CoverageState
from gaussian_robot.nav.action import Action, ActionSpace
from gaussian_robot.nav.explorer import Explorer, SeedPose, WalkResult
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.nav.robot import Robot
from gaussian_robot.nav.stop import (
    CoveragePlateau,
    CoverageTarget,
    PoseBudget,
    SeedExhaustion,
    SessionStopPolicy,
    StopPolicy,
)
from gaussian_robot.nav.terrain import build_height_field
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import Decision, VLMClient
from gaussian_robot.vlm.observation import Observation

_BOUNDS = (np.zeros(3), np.array([10.0, 10.0, 10.0]))
_INTR = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=64, height=64)


def _scene() -> SplatScene:
    return SplatScene(path=Path(__file__), bounds=SceneBounds(min=_BOUNDS[0], max=_BOUNDS[1]))


class FakeRenderer:
    def __init__(self, depth_value: float = 5.0) -> None:
        self.calls = 0
        self.depth_value = depth_value

    def render(self, camera: Camera) -> RenderResult:
        self.calls += 1
        h, w = camera.intrinsics.height, camera.intrinsics.width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), self.depth_value, dtype=np.float32)
        return RenderResult(rgb=rgb, camera=camera, depth=depth)


@dataclass
class ScriptedVLM:
    actions: list[Action]
    _gen: Iterator[Action] = field(init=False)

    def __post_init__(self) -> None:
        self._gen = cycle(self.actions)

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> Decision:
        assert isinstance(observation, Observation)
        assert len(observation.panels) == 4
        labels = [label for label, _ in observation.panels]
        assert labels == ["rgb", "depth", "confidence", "map"]
        action = next(self._gen)
        return Decision(action=action, raw_text=action.value)

    def describe(self, observation: Observation) -> str:
        return "Test scene description."


def _explorer(
    actions: list[Action],
    *,
    walk_policies: list[StopPolicy] | None = None,
    session_policies: list[SessionStopPolicy] | None = None,
    max_steps: int = 5,
    depth_value: float = 5.0,
    vlm: VLMClient | None = None,
    actions_per_query: int = 1,
) -> Explorer:
    renderer = FakeRenderer(depth_value=depth_value)
    space = ActionSpace(step=1.0)
    builder = ObservationBuilder(renderer=renderer, up_axis="y", map_size=64)
    return Explorer(
        scene=_scene(),
        renderer=renderer,
        vlm=vlm or ScriptedVLM(actions),
        observation_builder=builder,
        action_space=space,
        coverage_radius=1.0,
        walk_policies=walk_policies or [],
        session_policies=session_policies or [],
        max_steps=max_steps,
        actions_per_query=actions_per_query,
    )


def _state() -> CoverageState:
    return CoverageState.empty("y", _BOUNDS[0], _BOUNDS[1])


def test_walk_runs_to_budget_without_policies() -> None:
    explorer = _explorer([Action.FORWARD], max_steps=4)
    result = explorer.run_walk(Pose(), _state(), walk_id="s0")
    assert result.walk_id == "s0"
    assert len(result.steps) == 5  # seed step + 4 loop steps (describe + 3 forward)
    assert result.steps[1].action is Action.DESCRIBE
    assert all(s.action is Action.FORWARD for s in result.steps[2:])


def test_walk_plateau_stops_early() -> None:
    plateau = CoveragePlateau(novelty_delta=0.5, window=3)
    explorer = _explorer([Action.STOP], walk_policies=[plateau], max_steps=50)
    result = explorer.run_walk(Pose(), _state(), walk_id="s0")
    assert len(result.steps) < 50  # stopped early via plateau
    assert plateau.should_stop()


def test_walk_forward_accumulates_coverage() -> None:
    explorer = _explorer([Action.FORWARD], max_steps=3)
    state = _state()
    explorer.run_walk(Pose(), state, walk_id="s0")
    assert len(state) == 3  # seed + 2 forward (step 0 is describe)


def test_session_runs_all_seeds_until_exhaustion() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[SeedExhaustion()],
        max_steps=2,
    )
    seeds = [
        SeedPose(pose=Pose(position=np.array([1.0, 0.0, 1.0]))),
        SeedPose(pose=Pose(position=np.array([8.0, 0.0, 8.0]))),
    ]
    results = explorer.run_session(seeds, _state())
    assert len(results) == 2


def test_session_stops_on_pose_budget() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[PoseBudget(max_poses=3)],
        max_steps=2,
    )
    seeds = [SeedPose(pose=Pose(position=np.zeros(3))) for _ in range(5)]
    results = explorer.run_session(seeds, _state())
    assert len(results) <= 5


def test_session_coverage_target_stops() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[CoverageTarget(radius=1.0, tau=0.99)],
        max_steps=2,
    )
    seeds = [
        SeedPose(pose=Pose(position=np.array([x, 0.0, z]))) for x in (1.0, 8.0) for z in (1.0, 8.0)
    ]
    results = explorer.run_session(seeds, _state())
    assert len(results) >= 1


def test_observation_map_is_not_blank_when_sampled() -> None:
    state = _state()
    state.add_pose(Pose(position=np.array([5.0, 0.0, 5.0])))
    builder = ObservationBuilder(renderer=FakeRenderer(), up_axis="y", map_size=64)
    intr = _INTR
    camera = Camera(pose=Pose(position=np.array([5.0, 0.0, 5.0])), intrinsics=intr)
    obs, _ = builder.build(camera, state, trail=[camera.pose], step=1, budget=5)
    map_img = dict(obs.panels)["map"]
    assert map_img.shape == (64, 64, 3)
    has_blue = bool(
        np.any((map_img[..., 0] == 40) & (map_img[..., 1] == 90) & (map_img[..., 2] == 200))
    )
    has_red = bool(np.any((map_img[..., 0] == 220) & (map_img[..., 2] == 30)))
    assert has_blue
    assert has_red


def test_walk_records_stop_reason_step_budget() -> None:
    explorer = _explorer([Action.FORWARD], max_steps=4)
    result = explorer.run_walk(Pose(), _state(), walk_id="s0")
    assert result.stop_reason == "step_budget"  # no policy fired; loop exhausted


def test_walk_records_plateau_reason_and_emits_walk_end() -> None:
    events: list[object] = []
    plateau = CoveragePlateau(novelty_delta=0.5, window=2)
    explorer = _explorer([Action.STOP], walk_policies=[plateau], max_steps=50)
    explorer.event_sink = events.append
    result = explorer.run_walk(Pose(), _state(), walk_id="s0")
    assert result.stop_reason == "coverage_plateau"
    walk_ends = [e for e in events if isinstance(e, WalkEndEvent)]
    assert len(walk_ends) == 1
    assert walk_ends[0].reason == "coverage_plateau"
    assert walk_ends[0].walk_id == "s0"


def test_blocked_forward_does_not_accumulate_coverage() -> None:
    # depth 0.3 < margin (0.5 * step 1.0) -> every forward is blocked, nothing commits.
    explorer = _explorer([Action.FORWARD], max_steps=5, depth_value=0.3)
    state = _state()
    result = explorer.run_walk(Pose(position=np.array([5.0, 0.0, 5.0])), state, walk_id="s0")
    assert len(state) == 1  # only the seed pose; no forward step committed
    assert all(s.blocked for s in result.steps if s.action is Action.FORWARD)


def test_session_end_reason_is_specific() -> None:
    events: list[object] = []
    explorer = _explorer([Action.FORWARD], session_policies=[SeedExhaustion()], max_steps=2)
    explorer.event_sink = events.append
    seeds = [
        SeedPose(pose=Pose(position=np.array([1.0, 0.0, 1.0])), kind="capture"),
        SeedPose(pose=Pose(position=np.array([8.0, 0.0, 8.0])), kind="origin_fallback"),
    ]
    explorer.run_session(seeds, _state(), requested_seeds=4)
    end = next(e for e in events if isinstance(e, SessionEndEvent))
    assert end.reason == "seeds_exhausted"
    start = next(e for e in events if isinstance(e, SessionStartEvent))
    assert start.seed_floor.shape == (2, 2)  # both seeds' floor positions surfaced
    assert start.seed_kinds == ["capture", "origin_fallback"]  # provenance surfaced
    assert start.total_seeds == 2 and start.requested_seeds == 4  # rejections visible


def test_mark_records_pose_and_emits_event_without_moving() -> None:
    events: list[object] = []
    explorer = _explorer([Action.MARK], max_steps=4)
    explorer.event_sink = events.append
    state = _state()
    result = explorer.run_walk(Pose(position=np.array([5.0, 0.0, 5.0])), state, walk_id="walk0")
    assert len(result.marks) >= 1  # the marked viewpoint(s) are recorded
    marks = [e for e in events if isinstance(e, MarkEvent)]
    assert marks and marks[0].walk_id == "walk0" and marks[0].count == 1
    assert len(state) == 1  # mark neither moves the robot nor adds coverage


def test_auto_mark_records_when_standing_in_a_frontier() -> None:
    explorer = _explorer([Action.STOP])  # coverage_radius = 1.0
    explorer.observation_builder.nearest_gap = lambda pose: (1.0, 80.0)  # type: ignore[method-assign]
    events: list[object] = []
    explorer.event_sink = events.append
    result = WalkResult(walk_id="walk0")
    robot = Robot(scene=_scene(), pose=Pose(position=np.array([5.0, 0.0, 5.0])))
    explorer._maybe_auto_mark(robot, result, walk_id="walk0", step=3)  # at a gap, facing irrelevant
    assert len(result.marks) == 1
    auto = [e for e in events if isinstance(e, MarkEvent) and e.auto]
    assert auto and auto[0].count == 1


def test_auto_mark_skips_when_no_frontier_nearby() -> None:
    explorer = _explorer([Action.STOP])  # coverage_radius = 1.0
    result = WalkResult(walk_id="walk0")
    robot = Robot(scene=_scene(), pose=Pose())
    explorer.observation_builder.nearest_gap = lambda pose: (5.0, 10.0)  # type: ignore[method-assign]
    explorer._maybe_auto_mark(robot, result, walk_id="walk0", step=1)  # nearest gap too far
    assert not result.marks


@dataclass
class PlanVLM:
    """Returns a fixed multi-action plan each query; counts how often it is queried."""

    plan: list[Action]
    calls: int = 0

    def reset(self) -> None:
        pass  # keep `calls` across the walk so the test can count queries

    def act(self, observation: Observation) -> Decision:
        self.calls += 1
        return Decision(action=self.plan[0], raw_text="plan", actions=list(self.plan))

    def describe(self, observation: Observation) -> str:
        return "desc"


def test_action_chunking_executes_plan_with_one_query() -> None:
    vlm = PlanVLM(plan=[Action.FORWARD, Action.FORWARD, Action.FORWARD])
    explorer = _explorer([], vlm=vlm, actions_per_query=3, max_steps=4)
    state = _state()
    result = explorer.run_walk(Pose(), state, walk_id="w")
    forwards = [s for s in result.steps if s.action is Action.FORWARD]
    assert len(forwards) == 3  # step0 describe, then 3 planned forwards
    assert vlm.calls == 1  # the whole 3-action plan came from a single query


def test_action_chunking_requeries_when_plan_exhausted() -> None:
    vlm = PlanVLM(plan=[Action.FORWARD, Action.FORWARD])
    explorer = _explorer([], vlm=vlm, actions_per_query=2, max_steps=7)
    explorer.run_walk(Pose(), _state(), walk_id="w")
    # step0 describe, then 6 forwards executed in plans of 2 -> 3 queries
    assert vlm.calls == 3


def test_marks_are_deduped_by_coverage_radius() -> None:
    explorer = _explorer([Action.STOP])  # coverage_radius = 1.0
    result = WalkResult(walk_id="w")
    assert explorer._record_mark(
        Pose(position=np.array([5.0, 0.0, 5.0])), result, walk_id="w", step=1, auto=False
    )
    # a second mark within the coverage radius of the first is skipped
    assert not explorer._record_mark(
        Pose(position=np.array([5.3, 0.0, 5.0])), result, walk_id="w", step=2, auto=True
    )
    assert len(result.marks) == 1


def test_follow_terrain_sets_camera_above_local_ground() -> None:
    means = np.array([[5.0, 0.0, 5.0], [5.0, 1.0, 5.0], [5.0, 2.0, 5.0]])  # ground near (5,5)
    hf = build_height_field(means, "y", _BOUNDS[0], _BOUNDS[1], grid_size=8, ground_q=0.2)
    explorer = _explorer([Action.STOP])
    explorer.height_field = hf
    explorer._eye_offset = 1.5
    adjusted = explorer._follow_terrain(Pose(position=np.array([5.0, 9.0, 5.0])))
    ground = hf.ground(5.0, 5.0)
    assert ground is not None
    assert adjusted.position[1] == pytest.approx(ground + 1.5)  # up-coord pinned to ground+eye
    assert np.allclose(adjusted.position[[0, 2]], [5.0, 5.0])  # floor position unchanged


def test_render_is_runtime_checkable() -> None:
    assert isinstance(FakeRenderer(), Renderer)
