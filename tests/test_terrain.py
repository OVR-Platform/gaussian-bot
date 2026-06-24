"""Ground-height field for terrain-following on non-flat scenes."""

from __future__ import annotations

import numpy as np
import pytest

from gaussian_robot.nav.terrain import aerial_target, build_height_field


def _column(x: float, z: float, ys: list[float]) -> np.ndarray:
    return np.array([[x, y, z] for y in ys], dtype=np.float64)


def test_ground_is_low_quantile_of_heights() -> None:
    means = _column(1.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
    hf = build_height_field(
        means, "y", np.zeros(3), np.array([10.0, 10.0, 10.0]), grid_size=4, ground_q=0.2
    )
    assert hf.ground(1.0, 1.0) == pytest.approx(np.quantile([0, 1, 2, 3, 4], 0.2))  # 0.8
    assert hf.ground(9.0, 9.0) is None  # empty cell -> unknown


def test_ground_uses_signed_up_axis() -> None:
    # up = -y: height = -y, so the ground is the low quantile of the negated heights.
    means = _column(1.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
    hf = build_height_field(
        means, "-y", np.zeros(3), np.array([10.0, 10.0, 10.0]), grid_size=4, ground_q=0.2
    )
    assert hf.ground(1.0, 1.0) == pytest.approx(np.quantile([0, -1, -2, -3, -4], 0.2))


def test_sparse_cell_is_left_unknown() -> None:
    # A dense floor cell next to a sparse edge cell with only a couple of stray gaussians:
    # the sparse cell must stay unknown so terrain-following doesn't trust its bogus ground.
    lo, hi = np.zeros(3), np.array([10.0, 10.0, 10.0])
    dense = _column(1.0, 1.0, list(np.linspace(0.0, 1.0, 40)))  # well-supported floor cell
    sparse = _column(9.0, 9.0, [5.0, 6.0])  # 2 stray floaters at the far edge
    hf = build_height_field(np.vstack([dense, sparse]), "y", lo, hi, grid_size=4)
    assert hf.ground(1.0, 1.0) is not None  # dense cell trusted
    assert hf.ground(9.0, 9.0) is None  # sparse edge cell left unknown (not a phantom slope)


def test_empty_means_gives_all_unknown() -> None:
    hf = build_height_field(
        np.empty((0, 3)), "y", np.zeros(3), np.array([10.0, 10.0, 10.0]), grid_size=4
    )
    assert hf.ground(5.0, 5.0) is None


def test_aerial_target_finds_tall_geometry() -> None:
    ground = np.array([[float(x), 0.0, float(z)] for x in range(5) for z in range(5)])
    tower = np.array([[3.0, h, 7.0] for h in (5.0, 6.0, 7.0, 8.0)])
    res = aerial_target(np.vstack([ground, tower]), "y")
    assert res is not None
    xz, survey_h = res
    assert np.allclose(xz, [3.0, 7.0], atol=1.0)  # vantage over the tower
    assert survey_h > 8.0  # above the highest geometry


def test_aerial_target_none_on_flat_scene() -> None:
    flat = np.array([[float(x), 0.0, float(z)] for x in range(5) for z in range(5)])
    assert aerial_target(flat, "y") is None
