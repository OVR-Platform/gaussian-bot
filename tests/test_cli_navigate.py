"""The ``gaussian-robot navigate`` CLI: guards, wiring, outputs (ADR-0012).

``build_session`` is monkeypatched with a protocol-shaped fake session, so these run
CPU-only with no torch/gsplat/vLLM: they pin how the CLI wires the instruction, the
goal policy, the recorder, and the outcome/exit-code semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from gaussian_robot import session as session_mod
from gaussian_robot.cli import app
from gaussian_robot.events import StepEvent
from gaussian_robot.nav.action import Action
from gaussian_robot.nav.stop import GoalReached
from gaussian_robot.render.camera import Pose
from gaussian_robot.vlm.client import Decision
from gaussian_robot.vlm.observation import Observation

runner = CliRunner()


def test_navigate_help_lists_the_flags() -> None:
    result = runner.invoke(app, ["navigate", "--help"])
    assert result.exit_code == 0
    for flag in ("--instruction", "--target-xyz", "--goal-eps", "--demo-vlm", "--start-vllm"):
        assert flag in result.output


def test_navigate_rejects_empty_instruction(tmp_path: Path) -> None:
    result = runner.invoke(app, ["navigate", str(tmp_path / "s.ply"), "--instruction", "   "])
    assert result.exit_code == 2


class _FakeExplorer:
    """Duck-typed Explorer: walks 3 scripted steps toward +x, then stops on the budget."""

    def __init__(self, stop_reason: str) -> None:
        self.observation_builder = SimpleNamespace(task_target=None)
        self.walk_policies: list[object] = []
        self.event_sink = None
        self.scene = SimpleNamespace(up_axis="y")
        self.renderer = SimpleNamespace()
        self._stop_reason = stop_reason

    def run_walk(self, seed: Pose, coverage: object, *, walk_id: str = "") -> SimpleNamespace:
        assert self.event_sink is not None
        for i in range(3):
            pose = Pose(position=np.array([0.5 * i, 0.0, 0.0]))
            self.event_sink(
                StepEvent(
                    walk_id=walk_id,
                    step=i,
                    budget=3,
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
            )
        return SimpleNamespace(stop_reason=self._stop_reason)


def _fake_session(stop_reason: str) -> tuple[_FakeExplorer, list[SimpleNamespace], object]:
    explorer = _FakeExplorer(stop_reason)
    seeds = [SimpleNamespace(pose=Pose(), kind="capture")]
    return explorer, seeds, object()


def test_navigate_wires_goal_policy_recorder_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explorer, seeds, coverage = _fake_session("goal_reached")
    seen_cfg: dict[str, object] = {}

    def fake_build_session(config: object) -> tuple[object, object, object]:
        seen_cfg["mode"] = config.mode  # type: ignore[attr-defined]
        seen_cfg["task_prompt"] = config.task_prompt  # type: ignore[attr-defined]
        seen_cfg["use_real_vlm"] = config.use_real_vlm  # type: ignore[attr-defined]
        return explorer, seeds, coverage

    monkeypatch.setattr(session_mod, "build_session", fake_build_session)

    result = runner.invoke(
        app,
        [
            "navigate",
            str(tmp_path / "s.ply"),
            "--instruction",
            "go to the blue mat",
            "--target-xyz",
            "1.0,0.0,0.0",
            "--goal-eps",
            "0.8",
            "--demo-vlm",
            "--out-dir",
            str(tmp_path / "ep"),
            "--no-gif",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen_cfg == {
        "mode": "task",
        "task_prompt": "go to the blue mat",
        "use_real_vlm": False,
    }
    # The goal policy was composed first and the bearing hint was set on the builder.
    assert isinstance(explorer.walk_policies[0], GoalReached)
    tgt = explorer.observation_builder.task_target
    assert tgt is not None and np.allclose(tgt, [1.0, 0.0, 0.0])

    payload = json.loads((tmp_path / "ep" / "episode.json").read_text())
    assert payload["success"] is True and payload["success_source"] == "geometric"
    assert payload["stop_reason"] == "goal_reached"
    assert len(payload["trajectory"]) == 3
    assert "success=" in result.output


def test_navigate_vlm_declared_success_and_failure_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without a target, a VLM stop is an (unverified) success → exit 0, source vlm_declared.
    explorer, seeds, coverage = _fake_session("task_complete")
    monkeypatch.setattr(session_mod, "build_session", lambda cfg: (explorer, seeds, coverage))
    ok = runner.invoke(
        app,
        [
            "navigate",
            str(tmp_path / "s.ply"),
            "--instruction",
            "find the desk",
            "--demo-vlm",
            "--out-dir",
            str(tmp_path / "a"),
            "--no-gif",
        ],
    )
    assert ok.exit_code == 0 and "vlm_declared" in ok.output

    # A guard stop is a failure → exit 3.
    explorer2, seeds2, coverage2 = _fake_session("stuck")
    monkeypatch.setattr(session_mod, "build_session", lambda cfg: (explorer2, seeds2, coverage2))
    ko = runner.invoke(
        app,
        [
            "navigate",
            str(tmp_path / "s.ply"),
            "--instruction",
            "find the desk",
            "--demo-vlm",
            "--out-dir",
            str(tmp_path / "b"),
            "--no-gif",
        ],
    )
    assert ko.exit_code == 3
    payload = json.loads((tmp_path / "b" / "episode.json").read_text())
    assert payload["success"] is False and payload["stop_reason"] == "stuck"


def test_navigate_rejects_malformed_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explorer, seeds, coverage = _fake_session("goal_reached")
    monkeypatch.setattr(session_mod, "build_session", lambda cfg: (explorer, seeds, coverage))
    result = runner.invoke(
        app,
        [
            "navigate",
            str(tmp_path / "s.ply"),
            "--instruction",
            "go",
            "--target-xyz",
            "1.0,2.0",
            "--demo-vlm",
        ],
    )
    assert result.exit_code == 2
    assert "x,y,z" in result.output
