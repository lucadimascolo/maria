from __future__ import annotations

import numpy as np
import pytest
from maria.atmosphere.process import AutoregressiveProcess
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan
from maria.utils.linalg import fast_psd_inverse


def test_fast_psd_inverse_raises_on_an_indefinite_matrix():
    """The inverse of [[1, 2], [2, 1]] is [[-1/3, 2/3], [2/3, -1/3]]; the Cholesky route cannot give it."""
    with pytest.raises(np.linalg.LinAlgError):
        fast_psd_inverse(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_singular_covariance_is_retried_with_more_jitter(monkeypatch):
    """If the covariance is singular at the smallest jitter, the next attempt must use a larger jitter."""
    original = AutoregressiveProcess.compute_covariance_matrices
    seen = []

    def compute_covariance_matrices(self):
        seen.append(self.jitter)
        if self.jitter < 1e-6:
            raise np.linalg.LinAlgError("singular (simulated)")
        return original(self)

    monkeypatch.setattr(AutoregressiveProcess, "compute_covariance_matrices", compute_covariance_matrices)
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), stare_plan(duration=10))
    assert all(process.jitter == 1e-6 for process in sim.obs_list[0].atmosphere.processes.values())
    assert seen[:2] == [1e-8, 1e-6]
