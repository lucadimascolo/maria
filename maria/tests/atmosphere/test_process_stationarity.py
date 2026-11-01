from __future__ import annotations

import numpy as np
import pytest
from maria.atmosphere.process import AutoregressiveProcess
from maria.functions import approximate_normalized_matern


@pytest.mark.parametrize("n_extrusion, n_cross_section", [(14, 2), (28, 3), (14, 30)])
def test_autoregressive_process_is_stationary(n_extrusion, n_cross_section):
    """The default regime: an outer scale (1 km) much larger than the simulated screen (10 m grid). Every
    realization must be a draw from the stationary process: unit point variance, the Matern covariance between the
    two ends of the screen, and the Matern variance of the mean of the screen (its largest-scale mode, which the
    autoregression forgets most slowly)."""
    res, r0 = 10.0, 1e3
    matern_kwargs = {"nu": 5 / 6, "r0": r0}
    cross_section = np.c_[res * np.arange(n_cross_section), np.zeros(n_cross_section)]
    extrusion = res * np.arange(n_extrusion)
    process = AutoregressiveProcess(
        cross_section=cross_section,
        extrusion=extrusion,
        callback=approximate_normalized_matern,
        callback_kwargs=matern_kwargs,
        jitter=1e-8,
    )
    process.compute_covariance_matrices()

    np.random.seed(0)
    n_realizations = 600
    values = np.zeros((n_realizations, n_extrusion, n_cross_section))
    for i in range(n_realizations):
        process.run()
        values[i] = np.asarray(process.values)

    points = np.stack(np.meshgrid(extrusion, cross_section[:, 0], indexing="ij"), axis=-1).reshape(-1, 2)
    distance = np.sqrt(np.square(points[:, None] - points[None]).sum(axis=-1))
    covariance = approximate_normalized_matern(distance.ravel(), **matern_kwargs).reshape(distance.shape)
    np.fill_diagonal(covariance, 1.0)

    # each of these is estimated to sqrt(2 / N) ~ 6% (or better), so the bounds are > 4 sigma
    assert values.var(axis=0).mean() == pytest.approx(1.0, abs=0.25)
    end_to_end = covariance[0, (n_extrusion - 1) * n_cross_section]
    assert np.mean(values[:, 0, 0] * values[:, -1, 0]) == pytest.approx(end_to_end, abs=0.25)
    assert values.mean(axis=(1, 2)).var() == pytest.approx(covariance.mean(), abs=0.25)
