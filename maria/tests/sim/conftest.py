from __future__ import annotations

import maria
import pytest
from maria.instrument import Band


@pytest.fixture(scope="session")
def simulate_map():
    """Builds a noiseless Simulation of a map without atmosphere, for a daisy scan around (150, -30) deg."""

    def simulate(m, instrument=None, primary_size=10, center=(150.0, -30.0), radius=0.03, duration=20, **sim_kwargs):
        if instrument is None:
            band = Band(center=150e9, width=10e9, NET_RJ=1e-5)
            instrument = maria.get_instrument(array={"n": 1, "bands": [band], "primary_size": primary_size})
        plan = maria.Plan.generate(
            start_time="2025-01-01T07:00:00",
            scan_type="daisy",
            scan_parameters={"radius": radius, "speed": radius / 3},
            duration=duration,
            sample_rate=10,
            scan_center=center,
            frame="ra/dec",
            jitter=0,
        )
        sim_kwargs = {"noise": False, "atmosphere": None, "progress_bars": False, **sim_kwargs}
        return maria.Simulation(instrument=instrument, plans=plan, site="cerro_toco", map=m, **sim_kwargs)

    return simulate
