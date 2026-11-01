from __future__ import annotations

import dask.array as da
import maria
import numpy as np
import pytest
from maria import Simulation
from maria.plan import get_plan

NEP = 1e-17  # W√s
NEP_PER_LOADING = 1e-17  # W√s per pW


@pytest.mark.parametrize("loading_pW, expected_NEP", [(0.0, NEP), (2.0, NEP + 2 * NEP_PER_LOADING), (-0.5, NEP)])
def test_nep_per_loading(loading_pW, expected_NEP):
    """
    NEP_per_loading adds NEP per pW of optical loading; negative (zero-mean sky) loading must not lower the NEP.
    """
    band = {"center": 150e9, "width": 30e9, "NEP": NEP, "NEP_per_loading": NEP_PER_LOADING, "knee": 0.0}
    instrument = maria.Instrument(arrays=[{"n": 10, "primary_size": 5, "field_of_view": 0.2, "bands": {"f150": band}}])
    sample_rate = 50.0
    plan = get_plan("ten_second_zenith_stare", duration=60, sample_rate=sample_rate)
    np.random.seed(1)
    sim = Simulation(instrument=instrument, plans=plan, site="llano_de_chajnantor", progress_bars=False)

    obs = sim.obs_list[0]
    obs.loading = {"map": da.asarray(np.full(obs.shape, loading_pW))}
    sim._simulate_noise(obs)

    noise_W = 1e-12 * np.asarray(obs.loading["noise"], dtype=float)
    measured_NEP = (noise_W.std(axis=-1) / np.sqrt(sample_rate)).mean()
    # 10 dets x 3000 samples: the relative error is about 0.4%
    assert np.abs(measured_NEP / expected_NEP - 1) < 0.03, f"measured / expected NEP = {measured_NEP / expected_NEP:.3f}"
