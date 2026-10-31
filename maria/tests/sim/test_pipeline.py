from __future__ import annotations

import os

import maria
import numpy as np
import pytest
from maria import Simulation, all_sites
from maria.instrument import test_instruments
from maria.mapping import MaximumLikelihoodMapper

here, this_filename = os.path.split(__file__)


# a fixed draw and a fixed date, so that the test does not depend on the run (e.g. the summer weather at some
# sites is outside the atmospheric spectrum tables)
START_TIME = "2026-01-15T12:00:00+00:00"
rng = np.random.default_rng(seed=0)
n_sims = 10
test_sites = rng.choice(a=all_sites, size=n_sims)
test_az = rng.uniform(low=0, high=360, size=n_sims)
test_el = rng.uniform(low=30, high=90, size=n_sims)


@pytest.mark.parametrize(
    "instrument,site,az,el",
    zip(rng.choice(test_instruments, size=n_sims), test_sites, test_az, test_el),
)
def test_pipeline(instrument, site, az, el):
    site = maria.get_site(site)

    plan = maria.Plan.generate(
        scan_type="daisy",
        scan_parameters={"radius": 0.5, "speed": 0.1},  # in degrees
        duration=60,  # in seconds
        sample_rate=15,  # in Hz
        scan_center=(az, el),
        frame="az/el",
        start_time=START_TIME,
    )

    sim = Simulation(
        instrument=instrument, site=site, plans=[plan], atmosphere="2d", cmb="generate", cmb_kwargs={"nside": 256}
    )

    tod = sim.run()[0]

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.data[field]).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    mapper = MaximumLikelihoodMapper(
        frame="ra/dec",
        tods=[tod],
    )

    mapper.fit(epochs=2, steps_per_epoch=25)

    assert mapper.map.weight.sum() > 0
