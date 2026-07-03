"""VLLMServerProcess.wait_ready: the blocking readiness loop the navigate CLI relies on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gaussian_robot.config import RunConfig
from gaussian_robot.vlm import server as server_mod
from gaussian_robot.vlm.server import VLLMServerProcess


def _proc(alive: bool = True) -> SimpleNamespace:
    return SimpleNamespace(poll=lambda: None if alive else 1, pid=4242)


def test_wait_ready_returns_true_once_the_endpoint_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = VLLMServerProcess()
    server._process = _proc()  # type: ignore[assignment]
    answers = iter([False, False, True])
    monkeypatch.setattr(server_mod, "_server_ready", lambda cfg, **kw: next(answers))
    monkeypatch.setattr(
        server_mod, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda s: None)
    )

    assert server.wait_ready(RunConfig(), timeout=30.0, poll=0.0) is True


def test_wait_ready_fails_fast_when_the_child_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    server = VLLMServerProcess()
    server._process = _proc(alive=False)  # type: ignore[assignment]
    monkeypatch.setattr(server_mod, "_server_ready", lambda cfg, **kw: False)

    assert server.wait_ready(RunConfig(), timeout=30.0, poll=0.0) is False


def test_wait_ready_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    server = VLLMServerProcess()
    server._process = _proc()  # type: ignore[assignment]
    monkeypatch.setattr(server_mod, "_server_ready", lambda cfg, **kw: False)
    ticks = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(
        server_mod, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda s: None)
    )

    assert server.wait_ready(RunConfig(), timeout=50.0, poll=0.0) is False
