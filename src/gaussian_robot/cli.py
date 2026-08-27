"""Command-line entry point.

``navigate`` is the supported goal-conditioned episode runner (ADR-0012).
Heavy dependencies (torch/gsplat) are imported lazily inside the commands
that need them.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich import print as rprint

from gaussian_robot import __version__
from gaussian_robot.ui.server import serve_dashboard

if TYPE_CHECKING:
    import numpy as np

    from gaussian_robot.config import RunConfig

app = typer.Typer(
    name="gaussian-robot",
    help="Navigate a robot inside a Gaussian Splat using VLM perception.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the package version."""
    rprint(f"gaussian-robot v{__version__}")


@app.command()
def info() -> None:
    """Show the current R&D pipeline status (what is / isn't wired up)."""
    rprint(
        "[bold]gaussian-robot[/bold] pipeline status:\n"
        "  [green]splat loader[/green]   : 3DGS PLY → GaussianCloud + AABB + density grid\n"
        "  [green]renderer[/green]       : gsplat (CUDA) with CPU point-splat fallback\n"
        "  [green]VLM client[/green]     : Qwen via vLLM (OpenAI-compatible; --start-vllm)\n"
        "  [green]explorer[/green]       : local controller + multi-seed (ADR-0003)\n"
        "  [green]navigate[/green]       : instruction-conditioned episode (ADR-0012)\n"
        "  [green]metrics[/green]        : Tier-1 floor + Tier-2 pose-space + Tier-3 3D gaps\n"
        "  [green]filters[/green]        : quality → dedup → budget (ADR-0008)"
    )


@app.command()
def ui(host: str = "0.0.0.0", port: int = 8787, start_vllm: bool = False) -> None:
    """Run the local dashboard."""
    rprint(f"[bold]dashboard[/bold] http://{host}:{port}")
    serve_dashboard(host, port, start_vllm=start_vllm)


def _parse_target(target_xyz: str | None) -> np.ndarray | None:
    """``'x,y,z'`` → world position, or ``None``; exits with code 2 on a malformed value."""
    import numpy as np  # noqa: PLC0415

    if target_xyz is None:
        return None
    try:
        target = np.array([float(v) for v in target_xyz.split(",")], dtype=np.float64)
        if target.shape != (3,):
            raise ValueError
    except ValueError as exc:
        rprint("[red]--target-xyz must be 'x,y,z' (three floats).[/red]")
        raise typer.Exit(code=2) from exc
    return target


def _start_vllm_blocking(config: RunConfig, timeout: float) -> None:
    """Spawn vLLM and block until its OpenAI endpoint answers; exit 1 on timeout/death."""
    from gaussian_robot.vlm.server import VLLMServerProcess  # noqa: PLC0415

    server = VLLMServerProcess()
    status = server.start(config)
    rprint(f"[bold]vllm[/bold]     starting pid={status.pid} → {config.vlm_base_url}")
    if not server.wait_ready(config, timeout=timeout):
        rprint(f"[red]vLLM did not become ready within {timeout:.0f}s (see data/vllm.log).[/red]")
        raise typer.Exit(code=1)
    rprint("[bold]vllm[/bold]     ready")


@app.command()
def navigate(
    scene: Annotated[Path, typer.Argument(help="Splat .ply to navigate (read-only).")],
    instruction: Annotated[
        str, typer.Option(help="Natural-language goal, e.g. 'go to the blue mat'.")
    ],
    poses: Annotated[
        Path | None,
        typer.Option(help="Capture poses (COLMAP dir / cameras.json). Auto-discovered if unset."),
    ] = None,
    device: Annotated[str, typer.Option(help="CUDA device.")] = "cuda:0",
    max_steps: Annotated[int, typer.Option(help="Step budget for the episode.")] = 60,
    target_xyz: Annotated[
        str | None,
        typer.Option(
            help="Optional goal world position 'x,y,z' (same frame as the PLY): enables the "
            "TARGET bearing hint and MEASURED success via the goal-reach policy."
        ),
    ] = None,
    goal_eps: Annotated[
        float, typer.Option(help="Success radius (floor-plane metres) around --target-xyz.")
    ] = 1.2,
    demo_vlm: Annotated[
        bool, typer.Option("--demo-vlm", help="Use the scripted demo VLM (no server needed).")
    ] = False,
    vlm_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible endpoint of an already-running VLM.")
    ] = None,
    start_vllm: Annotated[
        bool, typer.Option("--start-vllm", help="Spawn vLLM and block until it is ready.")
    ] = False,
    vllm_timeout: Annotated[
        float, typer.Option(help="Max seconds to wait for vLLM readiness.")
    ] = 900.0,
    out_dir: Annotated[
        Path, typer.Option(help="Where the trajectory JSON + GIF are written.")
    ] = Path("data/episodes"),
    gif: Annotated[bool, typer.Option("--gif/--no-gif", help="Render the episode GIF.")] = True,
) -> None:
    """Run one instruction-conditioned navigation episode against the real stack.

    The instruction rides in the Observation + prompt (task mode); the episode ends on
    measured goal arrival (with --target-xyz), on the VLM declaring completion, or on the
    safety guards (bounds/stuck/step budget). Outputs: trajectory JSON, a
    [robot view | top-down map] GIF, and a printed outcome with success provenance —
    'geometric' (measured) or 'vlm_declared' (unverified, no target available).
    """
    from gaussian_robot.config import load_config  # noqa: PLC0415
    from gaussian_robot.episode import (  # noqa: PLC0415
        EpisodeRecorder,
        finalize_outcome,
        render_episode_gif,
    )
    from gaussian_robot.nav.stop import GoalReached  # noqa: PLC0415

    if not instruction.strip():
        rprint("[red]--instruction must be a non-empty mission.[/red]")
        raise typer.Exit(code=2)

    overrides: dict[str, object] = {
        "mode": "task",
        "task_prompt": instruction,
        "ply_path": str(scene),
        "use_real_vlm": not demo_vlm,
        "max_steps": max_steps,
        "aerial_survey": False,
        "coverage_3d": False,
        "cuda_device": device,
        # Headless: the live-dashboard tween would render extra frames per step for no viewer.
        "live_tween_frames": 0,
    }
    if poses is not None:
        overrides["poses_path"] = str(poses)
    if vlm_url is not None:
        overrides["vlm_base_url"] = vlm_url
    config = load_config().overrides(overrides)

    if start_vllm:
        _start_vllm_blocking(config, vllm_timeout)

    from gaussian_robot.session import build_session  # noqa: PLC0415

    explorer, seeds, coverage = build_session(config)
    up_axis = explorer.scene.up_axis

    target = _parse_target(target_xyz)
    if target is not None:
        explorer.observation_builder.task_target = target
        explorer.observation_builder.target_hint_mode = os.environ.get(
            "GR_TARGET_HINT", "bearing")   # bearing | distance | off (ablation switch)
        if explorer.observation_builder.target_hint_mode == "off":
            explorer.observation_builder.target_hint = False
        # BEFORE TaskStop, so simultaneous arrival+stop reports the measured reason.
        explorer.walk_policies.insert(0, GoalReached(target=target, eps=goal_eps, up_axis=up_axis))

    recorder = EpisodeRecorder(instruction=instruction, up_axis=up_axis)
    if target is not None:
        recorder.record.target = [float(v) for v in target]
        recorder.record.goal_eps = goal_eps
    explorer.event_sink = recorder

    rprint(f"[bold]episode[/bold]  instruction={instruction!r} seed={seeds[0].kind}")
    result = explorer.run_walk(seeds[0].pose, coverage, walk_id="navigate")
    record = finalize_outcome(recorder.record, result.stop_reason)

    colour = "green" if record.success else "red"
    rprint(
        f"[bold]outcome[/bold]  success=[{colour}]{record.success}[/{colour}] "
        f"({record.success_source})  stop_reason={record.stop_reason}  steps={record.steps}"
    )
    traj_path = record.write_json(out_dir / "episode.json")
    rprint(f"[bold]traj[/bold]     {traj_path}")

    if gif:
        movie = render_episode_gif(
            record, out_dir / "episode.gif", renderer=explorer.renderer, device=device
        )
        rprint(f"[bold]gif[/bold]      {movie}")

    raise typer.Exit(code=0 if record.success else 3)


if __name__ == "__main__":
    app()
