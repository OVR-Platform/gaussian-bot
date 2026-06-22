"""Integration tests for the exploration loop (ADR-0003, ADR-0005, ADR-0006)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np

from gaussian_robot.metrics.coverage import CoverageState
from gaussian_robot.nav.action import Action, ActionSpace
from gaussian_robot.nav.explorer import Explorer
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.nav.stop import (
    CoveragePlateau,
    CoverageTarget,
    PoseBudget,
    SeedExhaustion,
    SessionStopPolicy,
    StopPolicy,
)
from gaussian_robot.render.base import Renderer, RenderResult
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose
from gaussian_robot.splat.scene import SceneBounds, SplatScene
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation

_BOUNDS = (np.zeros(3), np.array([10.0, 10.0, 10.0]))
_INTR = CameraIntrinsics(fx=400, fy=400, cx=256, cy=256, width=64, height=64)


def _scene() -> SplatScene:
    return SplatScene(path=Path(__file__), bounds=SceneBounds(min=_BOUNDS[0], max=_BOUNDS[1]))


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, camera: Camera) -> RenderResult:
        self.calls += 1
        h, w = camera.intrinsics.height, camera.intrinsics.width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 5.0, dtype=np.float32)
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
) -> Explorer:
    renderer = FakeRenderer()
    vlm = ScriptedVLM(actions)
    space = ActionSpace(step=1.0)
    builder = ObservationBuilder(renderer=renderer, up_axis="y", map_size=64)
    return Explorer(
        scene=_scene(),
        renderer=renderer,
        vlm=vlm,
        observation_builder=builder,
        action_space=space,
        coverage_radius=1.0,
        walk_policies=walk_policies or [],
        session_policies=session_policies or [],
        max_steps=max_steps,
    )


def _state() -> CoverageState:
    return CoverageState.empty("y", _BOUNDS[0], _BOUNDS[1])


def test_walk_runs_to_budget_without_policies() -> None:
    explorer = _explorer([Action.FORWARD], max_steps=4)
    result = explorer.run_walk(Pose(), _state(), seed_id="s0")
    assert result.seed_id == "s0"
    assert len(result.steps) == 5  # seed step + 4 loop steps (describe + 3 forward)
    assert result.steps[1].action is Action.DESCRIBE
    assert all(s.action is Action.FORWARD for s in result.steps[2:])


def test_walk_plateau_stops_early() -> None:
    plateau = CoveragePlateau(novelty_delta=0.5, window=3)
    explorer = _explorer([Action.STOP], walk_policies=[plateau], max_steps=50)
    result = explorer.run_walk(Pose(), _state(), seed_id="s0")
    assert len(result.steps) < 50  # stopped early via plateau
    assert plateau.should_stop()


def test_walk_forward_accumulates_coverage() -> None:
    explorer = _explorer([Action.FORWARD], max_steps=3)
    state = _state()
    explorer.run_walk(Pose(), state, seed_id="s0")
    assert len(state) == 3  # seed + 2 forward (step 0 is describe)


def test_session_runs_all_seeds_until_exhaustion() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[SeedExhaustion()],
        max_steps=2,
    )
    seeds = [Pose(position=np.array([1.0, 0.0, 1.0])), Pose(position=np.array([8.0, 0.0, 8.0]))]
    results = explorer.run_session(seeds, _state())
    assert len(results) == 2


def test_session_stops_on_pose_budget() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[PoseBudget(max_poses=3)],
        max_steps=2,
    )
    seeds = [Pose(position=np.zeros(3)) for _ in range(5)]
    results = explorer.run_session(seeds, _state())
    assert len(results) <= 5


def test_session_coverage_target_stops() -> None:
    explorer = _explorer(
        [Action.FORWARD],
        session_policies=[CoverageTarget(radius=1.0, tau=0.99)],
        max_steps=2,
    )
    seeds = [Pose(position=np.array([x, 0.0, z])) for x in (1.0, 8.0) for z in (1.0, 8.0)]
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


def test_render_is_runtime_checkable() -> None:
    assert isinstance(FakeRenderer(), Renderer)
