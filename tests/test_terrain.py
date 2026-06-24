"""Ground-height field for terrain-following on non-flat scenes."""

from __future__ import annotations

import numpy as np
import pytest

from gaussian_robot.nav.terrain import build_height_field


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


def test_empty_means_gives_all_unknown() -> None:
    hf = build_height_field(
        np.empty((0, 3)), "y", np.zeros(3), np.array([10.0, 10.0, 10.0]), grid_size=4
    )
    assert hf.ground(5.0, 5.0) is None
