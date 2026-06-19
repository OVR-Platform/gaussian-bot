"""Scene loaders.

These are deliberately thin. The heavy parsing (PLY gaussian cloud, ``.splat``
binary, trained-checkpoint formats) is deferred until we pick a renderer — the
format we need to read is dictated by whichever training pipeline produced the
scene (e.g. ``gsplat``, INerFStudio, SuperSplat export).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaussian_robot.splat.scene import SceneBounds, SplatScene

_VALID_SUFFIXES = {".ply", ".splat", ".ply.gz"}


def load_scene(path: str | Path, *, bounds: SceneBounds | None = None) -> SplatScene:
    """Create a :class:`SplatScene` handle for the file at ``path``.

    Parameters
    ----------
    path:
        Path to the reconstruction file (or a directory of checkpoints).
    bounds:
        Optional explicit navigable AABB. If omitted, bounds are left as a
        unit cube centred at the origin as a placeholder until a real loader
        derives them from the gaussian cloud.

    Notes
    -----
    This does **not** load gaussians onto the GPU. It only validates the path
    and returns a handle. A renderer implementation is responsible for reading
    the actual data.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scene not found: {p}")
    if p.is_file() and p.suffix.lower() not in _VALID_SUFFIXES:
        raise ValueError(
            f"unsupported scene file {p.suffix!r}; expected one of {sorted(_VALID_SUFFIXES)}"
        )
    if bounds is None:
        # placeholder bounds — replace once a real loader computes them
        bounds = SceneBounds(min=np.full(3, -0.5), max=np.full(3, 0.5))
    return SplatScene(path=p, bounds=bounds)
