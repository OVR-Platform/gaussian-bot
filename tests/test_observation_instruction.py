"""Observation carries the instruction as a structured field in task mode (ADR-0012)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.backends.demo import FakeRenderer
from gaussian_robot.metrics.coverage import CoverageState
from gaussian_robot.nav.observation import ObservationBuilder
from gaussian_robot.render.camera import Camera, CameraIntrinsics, Pose


def _build(mode: str, task: str) -> ObservationBuilder:
    return ObservationBuilder(
        renderer=FakeRenderer(), up_axis="y", map_size=64, map_span=4.0, task=task, mode=mode
    )


def _camera() -> Camera:
    intr = CameraIntrinsics(fx=32.0, fy=32.0, cx=32.0, cy=32.0, width=64, height=64)
    return Camera(pose=Pose(position=np.zeros(3)), intrinsics=intr)


def test_task_mode_observation_carries_the_instruction_structured() -> None:
    builder = _build("task", "go to the blue mat")
    coverage = CoverageState.empty("y", np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
    obs, _ = builder.build(_camera(), coverage, trail=[Pose()])
    assert obs.instruction == "go to the blue mat"
    assert "go to the blue mat" in obs.prompt  # still woven into the prompt for the VLM


def test_densify_mode_observation_has_no_instruction() -> None:
    builder = _build("densify", "")
    coverage = CoverageState.empty("y", np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
    obs, _ = builder.build(_camera(), coverage, trail=[Pose()])
    assert obs.instruction is None
