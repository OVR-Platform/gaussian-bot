"""Reconstruction-frontier detection: real holes vs. walls/void (ADR-0007)."""

from __future__ import annotations

import types

import numpy as np

from gaussian_robot.nav.observation import (
    _line_of_sight_clear,
    _rotation_streak,
    frontier_floor_positions,
    frontier_mask,
)


def test_frontier_is_empty_cell_adjacent_to_observed() -> None:
    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0  # one observed (dense) cell
    mask = frontier_mask(grid, observed_min=0.1)
    # the 4-neighbours of the observed cell are empty AND adjacent -> frontiers
    assert mask[1, 2] and mask[3, 2] and mask[2, 1] and mask[2, 3]
    # the observed cell itself is not under-sampled -> not a frontier
    assert not mask[2, 2]
    # a far corner touches no observed cell -> open void, not a frontier
    assert not mask[0, 0]


def test_frontier_includes_interior_undersampled_pocket() -> None:
    grid = np.full((5, 5), 0.6)  # well reconstructed everywhere...
    grid[2, 2] = 0.05  # ...except an interior under-sampled pocket (sparse, not empty)
    mask = frontier_mask(grid, observed_min=0.1, under_frac=0.5)  # under_max = 0.5*median(0.6)=0.3
    assert mask[2, 2]  # the weak interior cell is a frontier worth filling
    assert not mask[0, 0]  # a fully-reconstructed corner is not


def test_no_frontier_when_uniform() -> None:
    assert not frontier_mask(np.ones((4, 4))).any()  # all reconstructed, none under-sampled
    assert not frontier_mask(np.zeros((4, 4))).any()  # nothing reconstructed at all


def test_frontier_floor_positions_maps_cells_into_bounds() -> None:
    grid = np.zeros((4, 4))
    grid[1, 1] = 1.0  # one observed cell -> its empty neighbours are frontiers
    cloud = types.SimpleNamespace(
        density_grid=grid, density_bounds=(np.zeros(3), np.array([4.0, 4.0, 4.0]))
    )
    renderer = types.SimpleNamespace(cloud=cloud)
    pts = frontier_floor_positions(renderer)
    assert pts.shape[1] == 2 and pts.shape[0] >= 1
    assert (pts >= 0).all() and (pts <= 4).all()  # all map inside the density bounds


def test_frontier_floor_positions_empty_without_cloud() -> None:
    assert frontier_floor_positions(types.SimpleNamespace()).shape == (0, 2)


def test_line_of_sight_blocked_by_dense_cell() -> None:
    grid = np.zeros((10, 10))
    grid[5, :] = 1.0  # an occupied wall at x-bin 5
    bounds = (np.zeros(3), np.array([10.0, 10.0, 10.0]))
    # a path crossing x=5 is blocked; one that stays on the near side is clear
    assert not _line_of_sight_clear(np.array([1.0, 1.0]), np.array([9.0, 1.0]), grid, bounds, 0.9)
    assert _line_of_sight_clear(np.array([1.0, 1.0]), np.array([4.0, 1.0]), grid, bounds, 0.9)


def test_rotation_streak_counts_trailing_turns() -> None:
    assert _rotation_streak(None) == 0
    assert _rotation_streak(["forward"]) == 0
    assert _rotation_streak(["forward", "turn_left", "turn_right"]) == 2
    assert _rotation_streak(["turn_left", "turn_left", "forward"]) == 0  # trailing action moved
    assert _rotation_streak(["forward", "look_up", "turn_left", "turn_left"]) == 3
