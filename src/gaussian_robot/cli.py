"""Command-line entry point.

Tiny for now — real subcommands (render, navigate, eval) land once the
renderer and VLM backends are chosen.
"""

from __future__ import annotations

import typer
from rich import print as rprint

from gaussian_robot import __version__

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
        "  [green]splat loader[/green]   : scene + AABB (placeholder loader)\n"
        "  [yellow]renderer[/yellow]      : protocol only (gsplat / web / ? — not chosen)\n"
        "  [yellow]VLM client[/yellow]    : protocol only (Qwen3.5-9B on vLLM — not wired)\n"
        "  [green]explorer[/green]        : local controller + multi-seed (ADR-0003)\n"
        "  [green]metrics[/green]         : Tier-1 floor coverage + Tier-2 pose-space\n"
        "  [green]filters[/green]         : quality → dedup → budget (ADR-0008)\n"
        "  [yellow]diffusion GT[/yellow]  : not wired (Goal-A eval, ADR-0007)"
    )


if __name__ == "__main__":
    app()
