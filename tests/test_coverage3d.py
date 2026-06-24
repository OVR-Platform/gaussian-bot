"""3D coverage: occupied voxels a camera couldn't see (incl. occluded) are gaps."""

from __future__ import annotations

import numpy as np

from gaussian_robot.metrics.coverage3d import build_coverage3d
from gaussian_robot.session import _clip_to_material_band

_LO = np.zeros(3)
_HI = np.full(3, 8.0)


def _two_voxel_means() -> tuple[np.ndarray, np.ndarray]:
    # geometry in voxels (4,4,2) (near) and (4,4,5) (far), 2 gaussians each
    means = np.array([[4.5, 4.5, 2.5]] * 2 + [[4.5, 4.5, 5.5]] * 2, dtype=np.float64)
    return means, np.ones(means.shape[0])


def test_occlusion_far_voxel_behind_near_is_a_gap() -> None:
    means, opac = _two_voxel_means()
    cam_pos = np.array([[4.5, 4.5, 0.2]])  # in front, looking +z (OpenCV identity)
    cam_rot = np.eye(3)[None]
    fov = np.array([np.pi / 2])
    cov = build_coverage3d(means, opac, cam_pos, cam_rot, fov, fov, _LO, _HI, grid=8)
    assert cov.occupied[4, 4, 2] and cov.occupied[4, 4, 5]
    assert cov.seen[4, 4, 2]  # near face is visible
    assert cov.gap_mask[4, 4, 5]  # far voxel occluded by the near one -> a gap
    assert not cov.gap_mask[4, 4, 2]


def test_no_cameras_makes_all_occupied_gaps() -> None:
    means, opac = _two_voxel_means()
    cov = build_coverage3d(
        means,
        opac,
        np.empty((0, 3)),
        np.empty((0, 3, 3)),
        np.empty(0),
        np.empty(0),
        _LO,
        _HI,
        grid=8,
    )
    assert int(cov.gap_mask.sum()) == int(cov.occupied.sum()) == 2


def test_interior_of_solid_block_is_not_a_gap() -> None:
    # A 3x3x3 solid block of occupied voxels with no cameras: only the exposed
    # surface counts as a gap; the single fully-enclosed centre voxel does not.
    centres = np.array(
        [[x + 0.5, y + 0.5, z + 0.5] for x in (2, 3, 4) for y in (2, 3, 4) for z in (2, 3, 4)],
        dtype=np.float64,
    )
    means = np.repeat(centres, 2, axis=0)
    cov = build_coverage3d(
        means,
        np.ones(means.shape[0]),
        np.empty((0, 3)),
        np.empty((0, 3, 3)),
        np.empty(0),
        np.empty(0),
        _LO,
        _HI,
        grid=8,
    )
    assert int(cov.occupied.sum()) == 27  # the full 3x3x3 block
    assert cov.occupied[3, 3, 3] and not cov.gap_mask[3, 3, 3]  # enclosed centre is not a gap
    assert int(cov.gap_mask.sum()) == 26  # only the 26 exposed-surface voxels


def test_clip_to_material_band_drops_floaters_above_ceiling() -> None:
    # up = -y: "higher" is more negative y. Material sits around y in [0, -2];
    # a gap far above the material (y = -50) is a floater and must be clipped.
    means = np.array([[0.0, -float(k) * 0.02, 0.0] for k in range(100)], dtype=np.float64)
    gaps = np.array([[0.0, -1.0, 0.0], [0.0, -50.0, 0.0]], dtype=np.float64)
    kept = _clip_to_material_band(gaps, means, "-y")
    assert kept.shape == (1, 3)
    assert np.allclose(kept[0], [0.0, -1.0, 0.0])  # in-band gap survives, floater dropped


def test_gap_centers_and_nearest() -> None:
    means, opac = _two_voxel_means()
    cam_pos = np.array([[4.5, 4.5, 0.2]])
    cov = build_coverage3d(
        means,
        opac,
        cam_pos,
        np.eye(3)[None],
        np.array([np.pi / 2]),
        np.array([np.pi / 2]),
        _LO,
        _HI,
        grid=8,
    )
    centers = cov.gap_centers()
    assert centers.shape == (1, 3)
    assert np.allclose(centers[0], [4.5, 4.5, 5.5])  # centre of the far voxel
    nearest = cov.nearest_gap(np.array([4.5, 4.5, 8.0]))
    assert nearest is not None and np.allclose(nearest, [4.5, 4.5, 5.5])
