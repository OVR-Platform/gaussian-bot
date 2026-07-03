"""Episode recording, outcome semantics, and the replay GIF (ADR-0012).

All CPU: the recorder consumes synthetic events; the GIF renderer gets an injected
protocol-shaped fake renderer (per ADR-0001), so no gsplat/GPU is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gaussian_robot.episode import (
    EpisodeRecord,
    EpisodeRecorder,
    finalize_outcome,
    render_episode_gif,
)
from gaussian_robot.events import CarryEvent, StepEvent
from gaussian_robot.nav.action import Action
from gaussian_robot.render.base import RenderResult
from gaussian_robot.render.camera import Camera, Pose
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation


def _step_event(step: int, pos: tuple[float, float, float]) -> StepEvent:
    pose = Pose(position=np.array(pos, dtype=np.float64))
    return StepEvent(
        walk_id="navigate",
        step=step,
        budget=10,
        observation=Observation(),
        decision=Decision(action=Action.FORWARD),
        action=Action.FORWARD,
        pose=pose,
        novelty=0.1,
        degenerate=False,
        coverage_floor=0.0,
        coverage_pose_space=0.0,
        sampled_floor=np.empty((0, 2)),
        trail_floor=np.empty((0, 2)),
    )


def test_recorder_captures_trajectory_actions_and_carry() -> None:
    rec = EpisodeRecorder(instruction="go to the mat", up_axis="y")
    rec(_step_event(0, (0.0, 0.0, 0.0)))
    rec(_step_event(1, (0.5, 0.0, 0.0)))
    rec(
        CarryEvent(
            walk_id="navigate", step=2, floor=np.array([0.5, 0.0]), kind="grab", carrying=True
        )
    )

    r = rec.record
    assert r.instruction == "go to the mat"
    assert r.trajectory == [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    assert r.actions == ["forward", "forward"]
    assert len(r.forwards) == 2 and len(r.forwards[0]) == 3
    assert r.grabs == [[0.5, 0.0]] and r.drops == []


def test_outcome_goal_reached_is_measured_success() -> None:
    r = EpisodeRecord(instruction="x", up_axis="y", target=[1.0, 0.0, 0.0], goal_eps=1.0)
    r.trajectory = [[0.0, 0.0, 0.0]]
    out = finalize_outcome(r, "goal_reached")
    assert out.success is True and out.success_source == "geometric"
    assert out.steps == 1


def test_outcome_vlm_stop_without_target_is_declared_success() -> None:
    r = EpisodeRecord(instruction="x", up_axis="y")
    out = finalize_outcome(r, "task_complete")
    assert out.success is True and out.success_source == "vlm_declared"


def test_outcome_vlm_stop_outside_target_radius_is_failure() -> None:
    # With a target present, GoalReached would have fired on a true arrival — a bare
    # VLM stop therefore means it declared completion away from the goal.
    r = EpisodeRecord(instruction="x", up_axis="y", target=[9.0, 0.0, 9.0], goal_eps=1.0)
    out = finalize_outcome(r, "task_complete")
    assert out.success is False and out.success_source == "geometric"


def test_outcome_guard_stops_are_failures() -> None:
    for reason in ("step_budget", "bounds", "stuck"):
        out = finalize_outcome(EpisodeRecord(instruction="x", up_axis="y"), reason)
        assert out.success is False


def test_record_roundtrips_to_json(tmp_path: Path) -> None:
    r = EpisodeRecord(instruction="go", up_axis="-y", trajectory=[[0, 0, 0]], steps=1)
    path = r.write_json(tmp_path / "episode.json")
    payload = json.loads(path.read_text())
    assert payload["instruction"] == "go" and payload["trajectory"] == [[0, 0, 0]]


class _FlatRenderer:
    """Protocol-shaped fake: constant grey frame for any camera."""

    def render(self, camera: Camera) -> RenderResult:
        return RenderResult(rgb=np.full((32, 32, 3), 128, np.uint8), camera=camera)


def test_render_episode_gif_writes_movie_with_injected_renderer(tmp_path: Path) -> None:
    r = EpisodeRecord(
        instruction="go to the mat",
        up_axis="y",
        trajectory=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.5]],
        forwards=[[0.0, 0.0, 1.0]] * 3,
        target=[2.0, 0.0, 1.0],
        goal_eps=1.0,
        grabs=[[0.5, 0.0]],
    )
    out = render_episode_gif(r, tmp_path / "ep.gif", renderer=_FlatRenderer(), panel=64)
    assert out["ok"] is True
    assert Path(str(out["out_gif"])).exists()
    assert out["n_frames"] == 3


def test_render_episode_gif_refuses_short_trajectory(tmp_path: Path) -> None:
    r = EpisodeRecord(instruction="x", up_axis="y", trajectory=[[0, 0, 0]], forwards=[[0, 0, 1]])
    out = render_episode_gif(r, tmp_path / "ep.gif", renderer=_FlatRenderer())
    assert out["ok"] is False
