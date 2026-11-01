from __future__ import annotations

import numpy as np
import pytest
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan
from maria.units import Quantity


def set_uniform_wind(obs, wind_east: float, wind_north: float):
    """Impose a height-independent wind (toward east, toward north, in m/s) and rebuild the atmosphere."""
    atmosphere = obs.atmosphere
    levels = atmosphere.weather.data["levels"]
    n = np.shape(levels["wind_east"])[-1]
    levels["wind_east"] = Quantity(np.full(n, wind_east), "m/s")
    levels["wind_north"] = Quantity(np.full(n, wind_north), "m/s")
    levels["wind_speed"] = Quantity(np.full(n, np.hypot(wind_east, wind_north)), "m/s")
    atmosphere.timestep = None
    atmosphere.initialize(obs)
    return atmosphere


def xcorr_lag(a, b, dt, maxlag):
    """Return the lag tau such that b(t) ~ a(t - tau), with parabolic refinement of the peak."""
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n), 2 * n)
    cc = np.r_[cc[-maxlag:], cc[: maxlag + 1]]
    k = np.argmax(cc)
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (k - maxlag) * dt


def apparent_screen_velocity(wind_east, wind_north, az, seed=1):
    """Estimate the drift velocity (east, north) of the PWV screen seen by a staring array, from the
    cross-correlation lags between detectors and their positions on the layer (computed from az/el here)."""
    np.random.seed(seed)
    instrument = small_instrument(n=91, field_of_view=3, shape="circle")
    plan = stare_plan(az=az, el=60, duration=60)
    sim = atmosphere_sim(instrument, plan, atmosphere_kwargs={"layers": {"boundaries": [0, 600]}})
    obs = sim.obs_list[0]
    atmosphere = set_uniform_wind(obs, wind_east, wind_north)
    h = float(atmosphere.layers.h.values[0])

    np.random.seed(seed + 100)
    atmosphere.simulate_pwv(instrument=obs.instrument)
    y = np.asarray(atmosphere.zenith_scaled_pwv, dtype=float)
    az, el = np.asarray(atmosphere.coords.az, dtype=float), np.asarray(atmosphere.coords.el, dtype=float)
    d = h / np.tan(el)
    north, east = (d * np.cos(az)).mean(axis=1), (d * np.sin(az)).mean(axis=1)
    dt = atmosphere.timestep

    i0 = np.argmin(np.hypot(north - north.mean(), east - east.mean()))
    dy = np.diff(y, axis=1)
    taus = np.array([xcorr_lag(dy[i0], dy[j], dt, maxlag=int(6 / dt)) for j in range(len(y))])
    s_north, s_east = np.linalg.lstsq(np.c_[north - north[i0], east - east[i0]], taus, rcond=None)[0]
    s2 = s_north**2 + s_east**2
    return np.array([s_east / s2, s_north / s2])


@pytest.mark.parametrize(
    "wind_east, wind_north, az",
    [(10.0, 0.0, 10.0), (0.0, 10.0, 10.0), (7.07, 7.07, 200.0)],
)
def test_screen_drifts_downwind(wind_east, wind_north, az):
    """A frozen screen blown by the wind (u toward east, v toward north) must drift with velocity (u, v)."""
    v = apparent_screen_velocity(wind_east, wind_north, az)
    expected = np.array([wind_east, wind_north])
    angle_error = np.degrees(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(expected[1], expected[0])))))
    # the estimate scatters by ~4 deg and ~3% between seeds; the bug gives 90 or 180 deg
    assert abs(angle_error) < 20, f"apparent velocity (east, north) = {v}, expected {expected}"
    assert np.hypot(*v) == pytest.approx(10.0, rel=0.2)
