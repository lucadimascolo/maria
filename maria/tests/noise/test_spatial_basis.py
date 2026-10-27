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
