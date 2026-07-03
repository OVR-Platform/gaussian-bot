"""Command-line entry point.

``enhance`` is the supported splat-enhancement path (ADR-0011); ``navigate``
lands with the VLN work. Heavy dependencies (torch/gsplat/diffusers) are
imported lazily inside the commands that need them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from gaussian_robot import __version__
from gaussian_robot.ui.server import serve_dashboard

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


@app.command()
def ui(host: str = "0.0.0.0", port: int = 8787, start_vllm: bool = False) -> None:
    """Run the local dashboard."""
    rprint(f"[bold]dashboard[/bold] http://{host}:{port}")
    serve_dashboard(host, port, start_vllm=start_vllm)


@app.command()
def enhance(
    scene: Annotated[
        Path, typer.Argument(help="Source splat .ply (read-only; a NEW file is always written).")
    ],
    out: Annotated[Path, typer.Option(help="Destination .ply (must differ from SCENE).")],
    colmap: Annotated[
        Path, typer.Option(help="COLMAP sparse model dir (cameras.bin/images.bin), e.g. sparse/0.")
    ],
    images: Annotated[Path, typer.Option(help="Directory with the capture images.")],
    device: Annotated[str, typer.Option(help="CUDA device.")] = "cuda:0",
    downscale: Annotated[
        float, typer.Option(help="Working-resolution factor for renders/targets.")
    ] = 0.5,
    camera_id: Annotated[int, typer.Option(help="COLMAP camera id to use.")] = 1,
    seeds: Annotated[
        int, typer.Option(help="Robot exploration seeds (walks) used to mark gap poses.")
    ] = 6,
    max_steps: Annotated[int, typer.Option(help="Max robot steps per walk.")] = 40,
    filler: Annotated[
        str, typer.Option(help="View filler: 'difix' (generative) or 'geometric'.")
    ] = "difix",
    dtype: Annotated[
        str, typer.Option(help="Difix precision (float16 halves the ~8.4 GB fp32 peak).")
    ] = "float16",
    denoise_steps: Annotated[
        int,
        typer.Option(
            help="Difix denoise steps: 1 = published single-step; N>1 walks a descending τ-ladder."
        ),
    ] = 1,
    sdedit: Annotated[
        bool,
        typer.Option(
            "--sdedit/--no-sdedit",
            help="ArtiFixer opacity-mix latent init (requires --denoise-steps >= 2).",
        ),
    ] = False,
    progressive: Annotated[
        bool,
        typer.Option(
            "--progressive/--rounds-mode",
            help=(
                "Rounds-based fill (default: the measured-safe recipe) vs the faithful "
                "progressive Difix3D+ loop (geometry moves + densify — measured to regress on "
                "the office scene; the gate reverts it to a no-op there)."
            ),
        ),
    ] = False,
    steps: Annotated[int, typer.Option(help="Progressive steps (only with --progressive).")] = 12,
    iters_per_step: Annotated[int, typer.Option(help="Distill iters per progressive step.")] = 150,
    rounds: Annotated[int, typer.Option(help="Fill rounds (only with --rounds-mode).")] = 3,
    iters: Annotated[
        int, typer.Option(help="Distill iters per round (only with --rounds-mode).")
    ] = 300,
    aggressive: Annotated[
        bool,
        typer.Option(
            help="Unfreeze geometry + gate off: visible enhancement, no held-out guarantee."
        ),
    ] = False,
    gif: Annotated[
        bool, typer.Option("--gif/--no-gif", help="Render the before/after GIF.")
    ] = True,
    gif_path: Annotated[
        Path | None, typer.Option(help="GIF path (default: <out>_before_after.gif).")
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Write the run report as JSON here.")] = None,
) -> None:
    """Enhance under-observed regions of a splat: robot marks gaps, Difix fills, distill-back.

    The robot explores the scene (coverage mode) and marks under-observed viewpoints; the
    reference-conditioned Difix3D+ filler cleans those views and the distiller bakes them into a
    NEW ply. Held-out real views guard against regression (Δ-PSNR is printed); the source .ply is
    never touched. Pays off on SPARSE captures — on dense, well-covered scenes the fill measures
    ~0 gain (see docs/research/README.md).
    """
    if out.resolve() == scene.resolve():
        rprint("[red]--out must differ from SCENE: the source .ply is read-only.[/red]")
        raise typer.Exit(code=2)
    if sdedit and denoise_steps < 2:
        rprint("[red]--sdedit requires --denoise-steps >= 2 (single-step cannot denoise it).[/red]")
        raise typer.Exit(code=2)

    from gaussian_robot.enhance.before_after import render_before_after_gif  # noqa: PLC0415
    from gaussian_robot.enhance.explore_fill import explore_and_fill  # noqa: PLC0415

    rep = explore_and_fill(
        scene,
        colmap,
        images,
        out,
        device=device,
        camera_id=camera_id,
        downscale=downscale,
        num_seeds=seeds,
        max_steps=max_steps,
        filler=filler,
        filler_dtype=dtype,
        denoise_steps=denoise_steps,
        sdedit=sdedit,
        iters=iters,
        rounds=rounds,
        aggressive=aggressive,
        progressive=progressive,
        steps=steps,
        iters_per_step=iters_per_step,
    )

    fill = rep.fill
    rprint(
        f"[bold]explore[/bold]  seeds={rep.n_seeds} walks={rep.n_walks} "
        f"marks={rep.n_marks} up_axis={rep.up_axis}"
    )
    if fill is not None:
        delta = fill.psnr_after - fill.psnr_before
        colour = "green" if delta >= 0 else "red"
        rprint(
            f"[bold]fill[/bold]     filler={fill.filler} rounds={fill.rounds_run} "
            f"frames={fill.n_gap_poses} mask={fill.fill_mask_frac:.3f} delta={fill.fill_delta:.4f}"
        )
        rprint(
            f"[bold]guard[/bold]    held-out PSNR {fill.psnr_before:.3f} → {fill.psnr_after:.3f} dB "
            f"([{colour}]Δ {delta:+.3f}[/{colour}])  "
            f"gaussians {fill.n_gaussians_before} → {fill.n_gaussians_after}"
        )
        rprint(f"[bold]vram[/bold]     peak {fill.peak_vram_gb:.1f} GB (fill phase, 24 GB budget)")
    rprint(f"[bold]out[/bold]      {rep.out_ply}")

    gif_out: str | None = None
    if gif:
        target = gif_path or out.with_name(out.stem + "_before_after.gif")
        movie = render_before_after_gif(
            rep.trajectory,
            scene,
            rep.out_ply,
            target,
            device=device,
            up_axis=rep.up_axis,
            marks=rep.marks,
        )
        gif_out = str(target) if movie.get("ok") else None
        rprint(f"[bold]gif[/bold]      {movie}")

    if report is not None:
        from dataclasses import asdict  # noqa: PLC0415

        payload = {
            "scene": str(scene),
            "out_ply": rep.out_ply,
            "n_seeds": rep.n_seeds,
            "n_walks": rep.n_walks,
            "n_marks": rep.n_marks,
            "up_axis": rep.up_axis,
            "fill": asdict(fill) if fill is not None else None,
            "gif": gif_out,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2))
        rprint(f"[bold]report[/bold]   {report}")


if __name__ == "__main__":
    app()
