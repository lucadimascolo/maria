from __future__ import annotations

import maria
import pytest
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan


def test_weather_outside_the_spectrum_fails_early():
    """The median July weather at chiang_mai (~60 mm PWV) is beyond the am tables (50 mm): building the simulation
    must raise a clear error, rather than a generic interpolation error deep inside the run."""
    plan = stare_plan(duration=10, start_time="2026-07-15T06:00:00+00:00")
    with pytest.raises(ValueError, match="PWV"):
        atmosphere_sim(small_instrument(n=4, field_of_view=0.05), plan, site=maria.Site(region="chiang_mai"))


def test_weather_inside_the_spectrum_is_fine():
    plan = stare_plan(duration=10, start_time="2026-01-15T06:00:00+00:00")
    atmosphere_sim(small_instrument(n=4, field_of_view=0.05), plan, site=maria.Site(region="chiang_mai")).run()
