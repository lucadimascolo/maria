from __future__ import annotations

import arrow
import maria
import numpy as np
import pytest
from maria.atmosphere import Atmosphere
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan
from maria.units import Quantity


@pytest.mark.parametrize(
    "site_kwargs, weather_kwargs",
    [
        ({"weather_quantiles": {"humidity": 0.9}}, {"quantiles": {"humidity": 0.9}}),
        ({"seasonal": False}, {"seasonal": False}),
        ({"diurnal": False}, {"diurnal": False}),
    ],
)
def test_site_weather_options_reach_the_atmosphere(site_kwargs, weather_kwargs):
    """Explicit weather options on the Site must be used by the simulated atmosphere."""
    site = maria.Site(region="green_bank", altitude=825, **site_kwargs)
    plan = stare_plan(el=60, duration=10)
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), plan, site=site)
    weather = sim.obs_list[0].atmosphere.weather

    expected = maria.Weather(region="green_bank", time=arrow.get(plan.time.mean()), altitude=825, **weather_kwargs)
    default = maria.Weather(region="green_bank", time=arrow.get(plan.time.mean()), altitude=825)
    assert abs(expected.pwv.mm - default.pwv.mm) > 0.01  # the option matters at this time
    assert weather.pwv.mm == pytest.approx(expected.pwv.mm, rel=1e-9)


@pytest.mark.parametrize(
    "atmosphere_kwargs, expected",
    [
        ({}, 0.03),
        ({"layers": {"pwv_rms_frac": 0.01}}, 0.01),
    ],
)
def test_pwv_rms_frac_reaches_the_layers(atmosphere_kwargs, expected):
    instrument = small_instrument(n=4, field_of_view=0.05)
    sim = atmosphere_sim(instrument, stare_plan(duration=10), atmosphere_kwargs=atmosphere_kwargs)
    atmosphere = sim.obs_list[0].atmosphere
    total_rms = np.sqrt(np.sum(atmosphere.layers.pwv_rms**2))
    assert total_rms / atmosphere.weather.pwv.mm == pytest.approx(expected, rel=1e-6)
    assert atmosphere.pwv_rms_frac == expected


def test_pwv_rms_frac_argument_reaches_the_layers():
    """Atmosphere(pwv_rms_frac=...) set the attribute but not the layers on master."""
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), stare_plan(duration=10))
    obs = sim.obs_list[0]
    atmosphere = Atmosphere(
        model="2d",
        timestamp=obs.plan.time.mean(),
        region=obs.site.region,
        altitude=obs.site.altitude,
        pwv_rms_frac=0.2,
        disable_progress_bars=True,
    )
    atmosphere.initialize(obs)
    total_rms = np.sqrt(np.sum(atmosphere.layers.pwv_rms**2))
    assert total_rms / atmosphere.weather.pwv.mm == pytest.approx(0.2, rel=1e-6)
    assert atmosphere.pwv_rms_frac == 0.2


def test_low_wind_short_stare():
    """With the lowest wind the automatic timestep is several seconds: it must still leave enough samples to upsample a
    short observation."""
    site = maria.Site(region="green_bank", altitude=825, weather_quantiles={"wind_speed": 0.0})
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), stare_plan(duration=3), site=site)
    tod = sim.run()[0]
    assert np.isfinite(np.asarray(tod.data["atmosphere"])).all()


def test_low_wind_timestep_unchanged_when_long_enough():
    """Observations that already had enough samples on master keep master's automatic timestep exactly."""
    site = maria.Site(region="green_bank", altitude=825, weather_quantiles={"wind_speed": 0.0})
    sim = atmosphere_sim(small_instrument(n=4, field_of_view=0.05), stare_plan(duration=15), site=site)
    obs = sim.obs_list[0]
    atmosphere = obs.atmosphere
    min_fwhm = obs.instrument.dets.angular_fwhm(z=atmosphere.max_height).min()
    max_wind = Quantity((atmosphere.layers.wind_speed / atmosphere.layers.h).values, "rad/s").max()
    master_timestep = max(1e-1, (min_fwhm / max_wind).s)
    duration = np.ptp(obs.boresight.t)
    assert duration / 4 < master_timestep < duration / 3  # a cap at a quarter of the observation would change it
    assert atmosphere.timestep == master_timestep
