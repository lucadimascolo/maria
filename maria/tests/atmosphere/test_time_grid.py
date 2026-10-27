from __future__ import annotations

import numpy as np
import pytest
from maria.coords import Coordinates


@pytest.mark.parametrize("timestep", [0.1, 0.3, 0.5, 1.0])
def test_downsample_covers_the_span(timestep):
    """The atmosphere is simulated on a downsampled time grid and interpolated to the TOD, so the grid must cover the
    whole TOD, also when the span is a multiple of the timestep at unix-time magnitudes (rounding)."""
    t = 1768449600.0 + np.arange(600) / 10
    coords = Coordinates(t=t, phi=np.zeros(t.size), theta=np.full(t.size, 1.0), frame="az/el")
    ds_t = coords.downsample(timestep=timestep).t
    assert ds_t.min() <= t.min() and ds_t.max() >= t.max()
