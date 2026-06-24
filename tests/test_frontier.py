"""Reconstruction-frontier detection: real holes vs. walls/void (ADR-0007)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.observation import frontier_mask


def test_frontier_is_empty_cell_adjacent_to_observed() -> None:
    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0  # one observed (dense) cell
    mask = frontier_mask(grid, empty_max=0.02, observed_min=0.1)
    # the 4-neighbours of the observed cell are empty AND adjacent -> frontiers
    assert mask[1, 2] and mask[3, 2] and mask[2, 1] and mask[2, 3]
    # the observed cell itself is not empty -> not a frontier
    assert not mask[2, 2]
    # a far corner touches no observed cell -> open void, not a frontier
    assert not mask[0, 0]


def test_no_frontier_when_uniform() -> None:
    assert not frontier_mask(np.ones((4, 4))).any()  # all observed, nothing empty
    assert not frontier_mask(np.zeros((4, 4))).any()  # all empty, nothing observed
