from __future__ import annotations

import numpy as np
import pytest
from maria.utils import compute_diameter
from maria.utils.linalg import generate_spatial_basis, matern_five_halves


@pytest.mark.parametrize("aspect", [1.0, 0.2])
def test_spatial_basis_covariance(aspect):
    """
    The spatial basis of the correlated noise must reproduce the Matern covariance it is built from,
    also for elongated focal planes (the interpolation grid used to be transposed).
    """
    rng = np.random.default_rng(0)
    offsets = rng.uniform(size=(300, 2)) * np.array([1.0, aspect])
    scale = compute_diameter(offsets)

    B = generate_spatial_basis(offsets=offsets, k=5, n_side=16, scale=scale)

    D = np.sqrt(np.square(offsets[:, None] - offsets[None]).sum(axis=-1))
    truth = matern_five_halves(D / scale)
    rms_error = np.sqrt(np.mean(np.square(B @ B.T - truth)))
    assert rms_error < 0.05, f"rms(B B^T - Matern) = {rms_error:.3f}"


def test_matern_five_halves():
    """
    The Matern-5/2 covariance is (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r) (the linear term had sqrt(3)).
    """
    r = np.linspace(0, 5, 101)
    truth = (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    assert np.allclose(matern_five_halves(r), truth, rtol=1e-12, atol=0)

    # a valid covariance: positive semi-definite on random points
    x = np.random.default_rng(1).uniform(size=(200, 2))
    C = matern_five_halves(np.sqrt(np.square(x[:, None] - x[None]).sum(axis=-1)) / 0.3)
    assert np.linalg.eigvalsh(C).min() > -1e-10


@pytest.mark.parametrize("axis", [0, 1])
def test_spatial_basis_of_a_line(axis):
    """A focal plane along a line (no extent in one dimension) raised in the grid interpolation."""
    offsets = np.zeros((20, 2))
    offsets[:, axis] = np.radians(np.linspace(-0.5, 0.5, 20))
    basis = generate_spatial_basis(offsets, scale=np.radians(1.0))
    assert basis.shape == (20, 5) and np.isfinite(basis).all()
    assert np.allclose(basis[:, 0], basis[::-1, 0], rtol=1e-6)  # symmetric line, symmetric leading mode


# a 1 deg line along xi through the boresight, and a 0.02 deg diagonal line at (2, 1.5) deg
LINES = {"on-axis": ((0.0, 0.0), (1.0, 0.0), 1.0), "off-axis": ((2.0, 1.5), (0.6, 0.8), 0.02)}


@pytest.mark.parametrize("line", list(LINES))
def test_noise_of_a_line_array(line):
    """
    A focal plane along a line with more than 16 detectors raised in compute_diameter and in the spatial basis
    (a short line away from the boresight looked two-dimensional to a rank test, because of rounding).
    """
    import maria

    center, direction, length = LINES[line]
    offsets = np.array(center) + length * np.linspace(-0.5, 0.5, 20)[:, None] * np.array(direction)
    array = {"xi": list(offsets[:, 0]), "eta": list(offsets[:, 1]), "degrees": True, "primary_size": 5}
    array["bands"] = {"f150": {"center": 150e9, "width": 30e9, "NET_RJ": 100e-6, "knee": 1.0}}
    plan = maria.Plan.generate(scan_type="stare", duration=10, sample_rate=10, frame="az/el", scan_center=(0, 60))
    sim = maria.Simulation(maria.get_instrument(array=array), plans=plan, site="cerro_toco", progress_bars=False)
    noise = np.asarray(sim.run()[0].data["noise"])
    assert noise.shape == (20, 100) and np.isfinite(noise).all()


@pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)])
def test_diameter_of_a_line(direction):
    points = np.linspace(-2.0, 3.0, 40)[:, None] * np.array(direction)
    assert compute_diameter(points) == pytest.approx(5.0)


def test_diameter_of_a_short_line_away_from_the_origin():
    """A 0.02 deg diagonal line at (2, 1.5) deg (in radians) raised a QhullError."""
    points = np.radians((2.0, 1.5) + np.linspace(-0.01, 0.01, 40)[:, None] * np.array((0.6, 0.8)))
    assert compute_diameter(points) == pytest.approx(np.radians(0.02), rel=1e-9)


@pytest.mark.parametrize("offset", [0.0, 100.0])
def test_diameter_of_coplanar_3d_points(offset):
    """
    Points on a tilted plane in 3D (e.g. interferometer baselines on a sloped site) raised a QhullError, since only
    dimensions without extent were masked out.
    """
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(40, 2))
    points = offset + np.c_[xy, 0.1 * xy[:, 0] + 0.2 * xy[:, 1]]
    i, j = np.triu_indices(len(points), k=1)
    expected = np.sqrt(np.square(points[i] - points[j]).sum(axis=-1)).max()
    assert compute_diameter(points) == pytest.approx(expected, rel=1e-9)
