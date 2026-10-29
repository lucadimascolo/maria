from __future__ import annotations

import maria
import numpy as np
from maria.atmosphere import Atmosphere
from maria.instrument import Instrument

START_TIME = "2026-01-15T04:00:00+00:00"


def small_instrument(n: int, field_of_view: float):
    band = maria.get_band("m2/f093")
    band.gain_error = 0.0
    return Instrument(arrays=[{"n": n, "field_of_view": field_of_view, "bands": [band], "primary_size": 6}])


def atmosphere_obs(plan, wind_quantile=0.5, timestep=None, n=1, field_of_view=0.0):
    sim = maria.Simulation(
        instrument=small_instrument(n=n, field_of_view=field_of_view),
        plans=[plan],
        site="green_bank",
        atmosphere="2d",
        noise=False,
        progress_bars=False,
    )
    obs = sim.obs_list[0]
    obs.atmosphere = Atmosphere(
        model="2d",
        timestamp=obs.plan.time.mean(),
        region=obs.site.region,
        altitude=obs.site.altitude,
        weather_quantiles={"wind_speed": wind_quantile},
        timestep=timestep,
        disable_progress_bars=True,
    )
    obs.atmosphere.initialize(obs)
    return sim, obs


def sample_with_timestep(sim, obs, timestep: float):
    """Sample the current turbulence of the atmosphere with some timestep, and return the PWV of the TOD."""
    atmosphere = obs.atmosphere
    atmosphere.timestep = timestep
    atmosphere.boresight = obs.boresight.downsample(timestep=timestep)
    atmosphere.coords = atmosphere.boresight.broadcast(obs.instrument.dets.offsets, frame="az/el")
    for process in atmosphere.processes.values():  # the 2d layers have a constant wind
        process.vx, process.vy = (np.full(len(atmosphere.boresight.t), v[0]) for v in (process.vx, process.vy))
    sim._simulate_atmosphere(obs)
    return np.asarray(obs.zenith_scaled_pwv)[0]


def test_scanning_timestep_resolves_the_beam_crossing():
    """At low wind the timestep was set by the wind alone (3.75 s here), although a 1 deg/s scan sweeps the 6 m beam
    across the turbulence in ~0.07 s. Sampling the same turbulence with the automatic timestep and with a fine (0.02 s)
    one, the structure function of the PWV at a lag of 0.1 s must match (0.98-1.00 here). Master's timestep gives
    0.34 of it (over independent realizations)."""
    plan = maria.Plan.generate(
        start_time=START_TIME,
        duration=120,
        sample_rate=50,
        scan_type="back_and_forth",
        scan_parameters={"x_throw": 2, "speed": 1},
        scan_center=(10, 60),
        frame="az/el",
    )
    np.random.seed(1)
    sim, obs = atmosphere_obs(plan, wind_quantile=0.0)
    timestep = obs.atmosphere.timestep
    min_fwhm = obs.instrument.dets.angular_fwhm(z=obs.atmosphere.max_height).min().rad
    assert timestep <= 1.01 * min_fwhm / np.radians(1)

    ratios = []
    for seed in range(4):
        np.random.seed(100 + seed)
        for process in obs.atmosphere.processes.values():
            process.run()
            process.run = lambda: None  # keep this turbulence for both timesteps
        y, y_ref = sample_with_timestep(sim, obs, timestep), sample_with_timestep(sim, obs, 0.02)
        ratios.append(np.mean((y[5:] - y[:-5]) ** 2) / np.mean((y_ref[5:] - y_ref[:-5]) ** 2))
        for process in obs.atmosphere.processes.values():
            del process.run

    assert 0.8 < np.mean(ratios) < 1.2


def test_stare_timestep_is_set_by_the_wind():
    """A stare does not move the beams, so its timestep is the wind crossing time with the 0.1 s floor, as before."""
    plan = maria.Plan.generate(
        start_time=START_TIME,
        duration=60,
        sample_rate=10,
        scan_type="stare",
        scan_center=(10, 60),
        frame="az/el",
    )
    for wind_quantile in [0.0, 0.5]:
        np.random.seed(1)
        sim, obs = atmosphere_obs(plan, wind_quantile=wind_quantile, n=7, field_of_view=0.1)
        atmosphere = obs.atmosphere
        min_fwhm = obs.instrument.dets.angular_fwhm(z=atmosphere.max_height).min().rad
        max_wind = (atmosphere.layers.wind_speed / atmosphere.layers.h).max()
        assert atmosphere.timestep == max(0.1, min_fwhm / max_wind)


def stare_timestep(atmosphere, obs):
    min_fwhm = obs.instrument.dets.angular_fwhm(z=atmosphere.max_height).min().rad
    return max(0.1, min_fwhm / (atmosphere.layers.wind_speed / atmosphere.layers.h).max())


def test_scan_timestep_ignores_jumps_and_jitter():
    """The scan speed must be robust. With the largest speed between two samples, a single jump of the boresight
    (here between two concatenated stares 10 deg apart) sets the timestep to the sample interval of the TOD (0.01 s),
    and a 1e-3 deg pointing jitter at 100 Hz cuts it from 3.75 s to 0.14 s at low wind. A jump must not change the
    stare timestep (in az/el), and the jitter must not collapse it (2.4 s here), while a slew between the stares (5 s
    out of 125 s) must still be resolved."""
    stare = {"sample_rate": 100, "scan_type": "stare", "frame": "az/el"}
    first = maria.Plan.generate(start_time=START_TIME, duration=60, scan_center=(10, 60), **stare)
    second = maria.Plan.generate(start_time=first.time[-1] + 0.01, duration=60, scan_center=(20, 60), **stare)
    t_slew = np.arange(0.01, 5, 0.01)

    plans = {
        "jump": maria.Plan(
            time=np.r_[first.time, second.time],
            phi=np.r_[first.phi, second.phi],
            theta=np.r_[first.theta, second.theta],
            frame="az/el",
        ),
        "jitter": maria.Plan(
            time=first.time,
            phi=first.phi + np.radians(1e-3) * np.random.default_rng(0).standard_normal(len(first.time)),
            theta=first.theta + np.radians(1e-3) * np.random.default_rng(1).standard_normal(len(first.time)),
            frame="az/el",
        ),
        "slew": maria.Plan(  # 2 deg/s in azimuth, so 1 deg/s on the sky
            time=np.r_[first.time, first.time[-1] + t_slew, second.time + 5],
            phi=np.r_[first.phi, first.phi[-1] + np.radians(2) * t_slew, second.phi],
            theta=np.r_[first.theta, np.full(len(t_slew), first.theta[-1]), second.theta],
            frame="az/el",
        ),
    }

    for wind_quantile in [0.0, 0.5]:
        timesteps = {}
        for name, plan in plans.items():
            np.random.seed(1)
            sim, obs = atmosphere_obs(plan, wind_quantile=wind_quantile)
            timesteps[name] = (obs.atmosphere.timestep, stare_timestep(obs.atmosphere, obs))
            min_fwhm = obs.instrument.dets.angular_fwhm(z=obs.atmosphere.max_height).min().rad

        assert timesteps["jump"][0] == timesteps["jump"][1]
        assert timesteps["jitter"][0] >= 0.5 * timesteps["jitter"][1]
        assert timesteps["slew"][0] <= 1.01 * min_fwhm / np.radians(1)
