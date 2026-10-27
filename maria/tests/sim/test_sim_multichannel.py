from __future__ import annotations

import numpy as np
import pytest
from maria.map import ProjectionMap


def flat_map(nu, channel_values):
    """A spectrally flat K_RJ map stored with the given channels."""
    data = np.asarray(channel_values, dtype=float)[:, None, None] * np.ones((len(nu), 32, 32))
    return ProjectionMap(data=data, width=2.0, nu=nu, center=(150.0, -30.0), units="K_RJ")


@pytest.fixture(scope="module", params=[None, "2d"])
def map_loading(request, simulate_map):
    """The mean map loading (in pW) per band of one test/1deg observation, as a function of the map channels."""
    np.random.seed(0)
    sim = simulate_map(flat_map([150e9], [0.0]), instrument="test/1deg", radius=0.2, atmosphere=request.param)
    obs = sim.obs_list[0]
    obs.loading = {}
    if request.param:
        sim._simulate_atmosphere(obs)  # the map loading only needs the PWV along the line of sight
    bands = np.asarray(obs.instrument.dets.band_name)

    def loading(nu, channel_values):
        sim._initialize_map(flat_map(nu, channel_values))
        sim._sample_maps(obs)
        return {band: np.asarray(obs.loading["map"])[bands == band].mean() for band in np.unique(bands)}

    return loading


def test_multichannel_map(map_loading):
    """A spectrally flat sky gives the same loading however many channels store it, and splitting it into its
    channels gives loadings that add up to the total."""
    A, nu = 1e-2, [140e9, 160e9]
    single = map_loading([150e9], [A])
    total = map_loading(nu, [A, A])
    low = map_loading(nu, [A, 0])
    high = map_loading(nu, [0, A])
    many = map_loading([90e9, 150e9, 220e9], 3 * [A])
    for band in single:
        assert np.isclose(total[band], single[band], rtol=1e-3), (band, total[band] / single[band])
        assert np.isclose(many[band], single[band], rtol=1e-3), (band, many[band] / single[band])
        assert np.isclose(low[band] + high[band], total[band], rtol=1e-3)
    # f150 straddles the channel boundary at 150 GHz, so both channels contribute
    assert 0.3 < low["f150"] / total["f150"] < 0.7
    # f090 is entirely below 150 GHz, f220 entirely above it
    assert np.isclose(low["f090"], total["f090"], rtol=1e-3)
    assert np.isclose(high["f220"], total["f220"], rtol=1e-3)
