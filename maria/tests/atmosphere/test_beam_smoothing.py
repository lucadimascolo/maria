from __future__ import annotations

import numpy as np
from maria.instrument import Instrument
from maria.tests.atmosphere._utils import atmosphere_sim, stare_plan


def test_each_band_is_smoothed_with_its_own_beam():
    """A small dish with two bands has very different beams at the same layer (here 3.4 m and 1.2 m at f093 and
    f280). Co-located detectors of the two bands must see the layer smoothed by their own beam, so the narrow-beam
    band has more small-scale power. Smoothing both with the mean beam makes them identical."""
    instrument = Instrument(arrays=[{"n": 20, "field_of_view": 0.5, "bands": ["so/f093", "so/f280"], "primary_size": 0.42}])
    np.random.seed(0)
    sim = atmosphere_sim(
        instrument,
        stare_plan(duration=60, sample_rate=10),
        site="cerro_toco",
        atmosphere_kwargs={"layers": {"boundaries": [0, 600]}},
    )
    atmosphere = sim.obs_list[0].atmosphere
    is_f093 = instrument.dets.band_name == "so/f093"

    ratios = []
    for seed in range(8):
        np.random.seed(10 + seed)
        atmosphere.simulate_pwv(instrument=instrument)
        y = np.asarray(atmosphere.zenith_scaled_pwv)
        ratios.append(np.var(np.diff(y[~is_f093], axis=1)) / np.var(np.diff(y[is_f093], axis=1)))

    assert np.mean(ratios) > 1.2
