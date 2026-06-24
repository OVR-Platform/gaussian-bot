"""Reconstruction-frontier detection: real holes vs. walls/void (ADR-0007)."""

from __future__ import annotations

import numpy as np

from gaussian_robot.nav.observation import _rotation_streak, frontier_mask


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


def test_rotation_streak_counts_trailing_turns() -> None:
    assert _rotation_streak(None) == 0
    assert _rotation_streak(["forward"]) == 0
    assert _rotation_streak(["forward", "turn_left", "turn_right"]) == 2
    assert _rotation_streak(["turn_left", "turn_left", "forward"]) == 0  # trailing action moved
    assert _rotation_streak(["forward", "look_up", "turn_left", "turn_left"]) == 3
