from __future__ import annotations

import maria
import pytest


def test_unknown_atmosphere_kwargs_raise():
    """A misspelled key used to be ignored silently, giving the default atmosphere."""
    plan = maria.Plan.generate(scan_type="stare", duration=1, sample_rate=10, scan_center=(10, 60), frame="az/el")
    with pytest.raises(ValueError, match="pwv_rms_fraction"):
        maria.Simulation(
            instrument="test/1deg",
            plans=plan,
            site="cerro_toco",
            atmosphere="2d",
            atmosphere_kwargs={"pwv_rms_fraction": 0.01},
        )

    # they are only checked when an atmosphere is simulated
    maria.Simulation(instrument="test/1deg", plans=plan, site="cerro_toco", atmosphere_kwargs={"pwv_rms_fraction": 0.01})


def test_unknown_weather_overrides_raise():
    """Only the PWV can be overridden: other keys of atmosphere_kwargs["weather"] used to be ignored silently."""
    plan = maria.Plan.generate(scan_type="stare", duration=1, sample_rate=10, scan_center=(10, 60), frame="az/el")
    with pytest.raises(ValueError, match="temperature"):
        maria.Simulation(
            instrument="test/1deg",
            plans=plan,
            site="cerro_toco",
            atmosphere="2d",
            atmosphere_kwargs={"weather": {"pwv": 1.0, "temperature": 280}},
        )

    sim = maria.Simulation(
        instrument="test/1deg", plans=plan, site="cerro_toco", atmosphere="2d", atmosphere_kwargs={"weather": {"pwv": 1.0}}
    )
    assert sim.obs_list[0].atmosphere.weather.pwv.mm == pytest.approx(1.0)


def test_top_level_pwv_rms_frac_points_to_the_layers():
    """Master only honoured atmosphere_kwargs["layers"]["pwv_rms_frac"]; a top-level key was ignored silently."""
    plan = maria.Plan.generate(scan_type="stare", duration=1, sample_rate=10, scan_center=(10, 60), frame="az/el")
    with pytest.raises(ValueError, match="'layers': {'pwv_rms_frac'"):
        maria.Simulation(
            instrument="test/1deg",
            plans=plan,
            site="cerro_toco",
            atmosphere="2d",
            atmosphere_kwargs={"pwv_rms_frac": 0.01},
        )
