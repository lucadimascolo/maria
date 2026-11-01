from __future__ import annotations

from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan


def test_no_atmosphere_progress_bars_when_disabled(capfd):
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), stare_plan(duration=10))  # progress_bars=False
    sim.run()
    err = capfd.readouterr().err
    for bar in ["Constructing atmosphere", "Generating turbulence", "Sampling turbulence"]:
        assert bar not in err
