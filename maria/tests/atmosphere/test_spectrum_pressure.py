from __future__ import annotations

import arrow
import h5py
import maria
import numpy as np
import pytest
import scipy as sp
from maria.constants import k_B
from maria.io import fetch
from maria.spectrum import AtmosphericSpectrum
from maria.tests.atmosphere._utils import START_TIME, atmosphere_sim, small_instrument, stare_plan


def am_table_band_power(band, region, pressure_level, effective_temperature, pwv, elevation):
    """Band power (pW) integrated directly from the am table, multilinear in (pressure, T_eff, pwv, csc el)."""
    with h5py.File(fetch(f"atmosphere/spectra/am/v4/{region}.h5"), "r") as f:
        side_p = f["side_pressure_level"][:].astype(float)
        points = tuple(f[k][:].astype(float) for k in ["side_effective_temperature", "side_pwv", "side_csc_el"])
        nu = f["side_nu"][:].astype(float)
        passband = band.passband(nu)
        m = passband > 0
        TRJ = f["temperature_rayleigh_jeans"][..., m].astype(float)
    P = 1e12 * k_B * np.trapezoid(TRJ * passband[m], nu[m], axis=-1)
    x = (effective_temperature, pwv, 1 / np.sin(elevation))
    P_at_nodes = [float(sp.interpolate.RegularGridInterpolator(points, P[i])(x)) for i in range(len(side_p))]
    return float(sp.interpolate.interp1d(side_p, P_at_nodes, fill_value="extrapolate")(pressure_level))


def test_loading_uses_site_pressure():
    """At cerro_chajnantor (5600 m, 600 m above the region default) the loading must match the am table at the
    pressure level of the site at the plan time, and the TOD must carry the same spectrum."""
    site = maria.get_site("cerro_chajnantor")
    instrument = small_instrument(n=4, field_of_view=0.05, band="act/pa5/f150")
    np.random.seed(0)
    sim = atmosphere_sim(
        instrument, stare_plan(el=50, duration=10), site=site, atmosphere_kwargs={"layers": {"pwv_rms_frac": 0.0}}
    )
    atmosphere = sim.obs_list[0].atmosphere
    tod = sim.run(units="pW")[0]

    weather = maria.Weather(region=site.region, time=arrow.get(START_TIME), altitude=site.altitude.m)
    p_site = weather.pressure_level.hPa
    assert atmosphere.spectrum.pressure_level.hPa == pytest.approx(p_site, abs=0.5)
    assert tod.spectrum.pressure_level.hPa == pytest.approx(atmosphere.spectrum.pressure_level.hPa, abs=1e-6)

    band = instrument.dets.bands[0]
    expected = am_table_band_power(
        band,
        site.region,
        p_site,
        weather.effective_temperature.K,
        weather.pwv.mm,
        float(np.median(tod.el)),
    )
    measured = float(np.median(np.asarray(tod.data["atmosphere"])))
    assert measured == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("region", ["chajnantor", "south_pole"])
def test_default_spectrum_does_not_depend_on_run_date(region, monkeypatch):
    levels = []
    for month in [1, 4, 7, 10]:
        fake_now = arrow.get(f"2026-{month:02d}-15T12:00:00+00:00")
        monkeypatch.setattr(arrow, "now", lambda *args, **kwargs: fake_now)
        levels.append(AtmosphericSpectrum(region=region).pressure_level.hPa)
    assert np.ptp(levels) < 1e-6


def am_table_transmission(region, pressure_level, nu, pwv, elevation):
    """exp(-opacity) at one frequency, multilinear in (pressure, T_eff, pwv, csc el, nu) directly from the am table,
    at the median effective temperature of the table (the spectrum's default)."""
    with h5py.File(fetch(f"atmosphere/spectra/am/v4/{region}.h5"), "r") as f:
        side_p = f["side_pressure_level"][:].astype(float)
        side_T, side_pwv, side_csc = (
            f[k][:].astype(float) for k in ["side_effective_temperature", "side_pwv", "side_csc_el"]
        )
        side_nu = f["side_nu"][:].astype(float)
        i = np.searchsorted(side_nu, nu) - 1
        opacity = f["opacity"][..., i : i + 2].astype(float)
    x = (pressure_level, np.median(side_T), pwv, 1 / np.sin(elevation), nu)
    points = (side_p, side_T, side_pwv, side_csc, side_nu[i : i + 2])
    return float(np.exp(-sp.interpolate.RegularGridInterpolator(points, opacity)(x)))


def test_band_transmission():
    band = maria.get_band("act/pa5/f150")
    nu, el = band.center.Hz, np.radians(45)
    default_p = AtmosphericSpectrum(region="chajnantor").pressure_level.hPa
    assert band.transmission(pwv=2, elevation=el) == pytest.approx(
        am_table_transmission("chajnantor", default_p, nu, 2, el), rel=1e-4
    )
    assert band.spectrum is None  # computing a transmission does not give the band a spectrum (used for its NET)

    # a band with its own spectrum (for its NET) uses it for its region, and keeps it for another region
    site_band = maria.band.Band(center=nu, width=30e9, NET_RJ=1e-5, spectrum_kwargs={"region": "chajnantor"})
    spectrum = site_band.spectrum
    assert site_band.transmission(pwv=2, elevation=el) == pytest.approx(
        am_table_transmission("chajnantor", default_p, nu, 2, el), rel=1e-4
    )
    site_band.transmission(region="green_bank", pwv=2, elevation=el)
    assert site_band.spectrum is spectrum
