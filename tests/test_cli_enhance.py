"""The ``gaussian-robot enhance`` CLI: guards, plumbing, and report output (ADR-0011).

The guard tests run before any heavy import, so they exercise the CLI in a CPU/no-torch CI
environment. The plumbing test stubs ``explore_and_fill`` / ``render_before_after_gif`` at
their modules (the command binds them at call time) and checks the flags arrive intact and
the JSON report is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gaussian_robot.cli import app

runner = CliRunner()


def test_enhance_help_lists_the_supported_flags() -> None:
    result = runner.invoke(app, ["enhance", "--help"])
    assert result.exit_code == 0
    for flag in ("--out", "--colmap", "--images", "--denoise-steps", "--sdedit", "--gif"):
        assert flag in result.output


def test_enhance_refuses_to_overwrite_the_source_ply(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    result = runner.invoke(
        app,
        [
            "enhance",
            str(ply),
            "--out",
            str(ply),
            "--colmap",
            str(tmp_path),
            "--images",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "read-only" in result.output


def test_enhance_rejects_sdedit_without_multi_step(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "enhance",
            str(tmp_path / "a.ply"),
            "--out",
            str(tmp_path / "b.ply"),
            "--colmap",
            str(tmp_path),
            "--images",
            str(tmp_path),
            "--sdedit",
            "--denoise-steps",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "--denoise-steps >= 2" in result.output


def test_enhance_plumbs_flags_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")  # explore_fill -> orchestrator imports torch + gsplat
    pytest.importorskip("gsplat")
    from gaussian_robot.enhance import before_after, explore_fill  # noqa: PLC0415
    from gaussian_robot.enhance.explore_fill import ExploreFillReport  # noqa: PLC0415
    from gaussian_robot.enhance.orchestrator import FillReport  # noqa: PLC0415

    seen: dict[str, object] = {}

    def _fake_explore_and_fill(*args: object, **kw: object) -> ExploreFillReport:
        seen.update(kw)
        fill = FillReport(
            n_views=24,
            n_anchor=20,
            n_eval=2,
            gap_count=5,
            n_gap_poses=3,
            n_gaussians_before=100,
            n_gaussians_after=110,
            filler="difix",
            psnr_before=28.0,
            psnr_after=28.4,
            out_ply=str(tmp_path / "b.ply"),
            peak_vram_gb=9.5,
        )
        return ExploreFillReport(
            n_seeds=2,
            n_walks=2,
            n_marks=4,
            up_axis="y",
            fill=fill,
            out_ply=fill.out_ply,
        )

    gif_calls: list[str] = []

    def _fake_gif(*args: object, **kw: object) -> dict[str, object]:
        gif_calls.append(str(args[3]))
        return {"ok": True, "n_frames": 10}

    monkeypatch.setattr(explore_fill, "explore_and_fill", _fake_explore_and_fill)
    monkeypatch.setattr(before_after, "render_before_after_gif", _fake_gif)

    report = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "enhance",
            str(tmp_path / "a.ply"),
            "--out",
            str(tmp_path / "b.ply"),
            "--colmap",
            str(tmp_path / "sparse"),
            "--images",
            str(tmp_path / "images"),
            "--denoise-steps",
            "2",
            "--sdedit",
            "--dtype",
            "float16",
            "--seeds",
            "3",
            "--rounds-mode",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["denoise_steps"] == 2 and seen["sdedit"] is True
    assert seen["filler_dtype"] == "float16"
    assert seen["num_seeds"] == 3
    assert seen["progressive"] is False  # --rounds selected the legacy path
    assert gif_calls, "the before/after GIF must be rendered by default"

    payload = json.loads(report.read_text())
    assert payload["fill"]["psnr_after"] == 28.4
    assert payload["fill"]["peak_vram_gb"] == 9.5
    assert payload["n_marks"] == 4
    assert "28.4" in result.output  # Δ-PSNR guard is printed
