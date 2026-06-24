"""Deliverable assembly prefers VLM-marked fill-in poses (ADR-0008)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.action import Action
from gaussian_robot.nav.explorer import WalkResult, WalkStep
from gaussian_robot.render.camera import Pose
from gaussian_robot.session import assemble_deliverable


def _walk_with(*, marks: list[Pose], steps: list[Pose]) -> WalkResult:
    r = WalkResult(walk_id="walk0")
    r.marks = marks
    r.steps = [
        WalkStep(pose=p, action=Action.FORWARD, novelty=1.0, degenerate=False) for p in steps
    ]
    return r


def test_deliverable_prefers_marks() -> None:
    r = _walk_with(
        marks=[Pose(position=np.array([9.0, 0.0, 9.0]))],
        steps=[Pose(position=np.array([0.0, 0.0, 0.0]))],
    )
    out = assemble_deliverable([r], up_axis="y", r_keep=0.0, budget=10)
    assert len(out) == 1
    assert np.allclose(out[0].pose.position, [9.0, 0.0, 9.0])  # the mark, not the trajectory


def test_deliverable_falls_back_to_trajectory_when_no_marks() -> None:
    r = _walk_with(marks=[], steps=[Pose(position=np.array([1.0, 0.0, 2.0]))])
    out = assemble_deliverable([r], up_axis="y", r_keep=0.0, budget=10)
    assert len(out) == 1
    assert np.allclose(out[0].pose.position, [1.0, 0.0, 2.0])
