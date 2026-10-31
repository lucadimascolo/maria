from __future__ import annotations

import maria
from maria.instrument import Instrument

START_TIME = "2026-01-15T04:00:00+00:00"


def stare_plan(az=10.0, el=60.0, duration=60.0, sample_rate=10.0, start_time=START_TIME, **kwargs):
    return maria.Plan.generate(
        start_time=start_time,
        duration=duration,
        sample_rate=sample_rate,
        scan_type="stare",
        scan_parameters={},
        scan_center=(az, el),
        frame="az/el",
        **kwargs,
    )


def small_instrument(n: int = 7, field_of_view: float = 0.1, band: str = "m2/f093", **kwargs):
    b = maria.get_band(band)
    b.gain_error = 0.0
    return Instrument(arrays=[{"n": n, "field_of_view": field_of_view, "bands": [b], "primary_size": 6, **kwargs}])


def atmosphere_sim(instrument, plan, site="green_bank", atmosphere="2d", **kwargs):
    return maria.Simulation(
        instrument=instrument,
        plans=[plan],
        site=site,
        atmosphere=atmosphere,
        noise=False,
        progress_bars=False,
        **kwargs,
    )
