from __future__ import annotations

import numpy as np
import pytest
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan


@pytest.mark.parametrize("az, el, roll", [(10.0, 50.0, 0.0), (10.0, 50.0, 37.0), (200.0, 80.0, 90.0), (300.0, 30.0, -60.0)])
def test_atmosphere_samples_the_rolled_focal_plane(az, el, roll):
    """The atmosphere must sample each detector at the same (az, el) as the rest of the simulation."""
    instrument = small_instrument(n=37, field_of_view=1.0)
    sim = atmosphere_sim(instrument, stare_plan(az=az, el=el, duration=10, roll=roll))
    obs = sim.obs_list[0]
    atmosphere = obs.atmosphere

    # a stare: the per-detector pointing is constant in time, so compare the time averages
    for attr in ["az", "el"]:
        atm = np.unwrap(np.asarray(getattr(atmosphere.coords, attr), dtype=float), axis=0).mean(axis=-1)
        sim_ = np.unwrap(np.asarray(getattr(obs.coords, attr), dtype=float), axis=0).mean(axis=-1)
        assert np.degrees(np.abs(atm - sim_)).max() * 3600 < 1.0  # arcsec
