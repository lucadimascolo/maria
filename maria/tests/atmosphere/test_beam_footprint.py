from __future__ import annotations

import jax
import maria
import numpy as np
import scipy as sp
from maria.atmosphere import Atmosphere
from maria.coords.transforms import unjitted_offsets_to_phi_theta
from maria.instrument import Instrument

START_TIME = "2026-01-15T04:00:00+00:00"


def windless_layer(field_of_view: float, elevation: float, n: int = 19):
    """A single turbulent layer (at 300 m) without wind, scanned in azimuth by a small array of detectors."""
    band = maria.get_band("m2/f093")
    band.gain_error = 0.0
    instrument = Instrument(arrays=[{"n": n, "field_of_view": field_of_view, "bands": [band], "primary_size": 6.0}])
    plan = maria.Plan.generate(
        start_time=START_TIME,
        duration=30,
        sample_rate=10,
        scan_type="back_and_forth",
        scan_parameters={"x_throw": 5.0, "speed": 0.5},
        scan_center=(10, elevation),
        frame="az/el",
    )
    np.random.seed(0)
    sim = maria.Simulation(instrument=instrument, plans=[plan], site="green_bank", noise=False, progress_bars=False)
    obs = sim.obs_list[0]
    atmosphere = Atmosphere(
        model="2d",
        timestamp=obs.plan.time.mean(),
        region=obs.site.region,
        altitude=obs.site.altitude,
        kwargs={"layers": {"boundaries": [0.0, 600.0]}},
        timestep=0.1,
        disable_progress_bars=True,
    )
    for field in ["wind_speed", "wind_east", "wind_north"]:
        atmosphere.weather.data["levels"][field] *= 0
    atmosphere.initialize(obs)
    return instrument, atmosphere


def sample_screen(instrument, atmosphere, seed: int, res: float = 0.25, n_quad: int = 12):
    """
    Replace the turbulence of the layer by a finely sampled Kolmogorov-like screen, and return the PWV fluctuations
    that the atmosphere samples (y_sim), the truth (y_true: the screen averaged over the beam of each detector, ray by
    ray, as it intersects the layer), and the unsmoothed screen at the center of each beam (y_raw).
    """
    (process,) = atmosphere.processes.values()
    layer = process.layers.iloc[0]
    extrusion, cross_section = process.extrusion, process.cross_section[:, 0]

    tx = np.arange(extrusion.min() - 40, extrusion.max() + 40, res)
    ty = np.arange(cross_section.min() - 40, cross_section.max() + 40, res)
    k2 = (2 * np.pi * np.fft.fftfreq(len(tx), res))[:, None] ** 2 + (2 * np.pi * np.fft.fftfreq(len(ty), res))[None] ** 2
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(k2.shape) + 1j * rng.standard_normal(k2.shape)
    screen = np.fft.ifft2((k2 + (2 * np.pi / 100) ** 2) ** (-11 / 12) * noise).real
    screen = sp.interpolate.RegularGridInterpolator((tx, ty), screen / screen.std())

    grid_tx, grid_ty = np.meshgrid(extrusion, cross_section, indexing="ij")
    process.values = screen(np.stack([grid_tx, grid_ty], axis=-1))
    process.run = lambda: None
    atmosphere.simulate_pwv(instrument=instrument)
    y_sim = (np.asarray(atmosphere.zenith_scaled_pwv) - atmosphere.weather.pwv.mm) / layer.pwv_rms

    def screen_along(az, el):
        p = layer.h * np.stack([np.cos(az) / np.tan(el), np.sin(az) / np.tan(el), np.ones(az.shape)], axis=-1)
        return screen((p @ process.transform)[..., :2])

    az = np.asarray(atmosphere.coords.az, dtype=float)
    el = np.asarray(atmosphere.coords.el, dtype=float)
    # rays from the beam center: the transverse width of the beam is that of the layer (as in the atmosphere)
    sigma = (instrument.dets.physical_fwhm(layer.z).m / 2.355)[:, None] * np.sin(el) / layer.h
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    weights /= weights.sum()

    def ray(dx, dy):
        # the exact (spherical) offsets: near the zenith, linearized az/el offsets put a 2-sigma ray up to 1.45 m
        # off on the layer. jax computes in float32 by default, so these are computed in float64.
        enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
        with enable_x64(True):
            ray_az, ray_el = np.moveaxis(np.asarray(unjitted_offsets_to_phi_theta(np.stack([dx, dy], -1), az, el)), -1, 0)
        assert ray_el.dtype == np.float64
        return ray_az, ray_el

    y_true = sum(
        wi * wj * screen_along(*ray(-sigma * xi, sigma * xj))  # (+x is towards decreasing azimuth)
        for xi, wi in zip(nodes, weights)
        for xj, wj in zip(nodes, weights)
    )

    return y_sim, y_true, screen_along(az, el)


def smoothing_statistics(instrument, atmosphere, n_seeds: int = 4):
    """
    The beam-scale power removed by the smoothing, relative to the truth (1 when correct), and the variance of the
    error relative to the beam-scale power.
    """
    removed, error = [], []
    for seed in range(n_seeds):
        y_sim, y_true, y_raw = sample_screen(instrument, atmosphere, seed=seed)
        removed.append(np.var(y_sim - y_raw) / np.var(y_true - y_raw))
        error.append(np.var(y_sim - y_true) / np.var(y_true - y_raw))
    return np.mean(removed), np.mean(error)


def test_cross_section_spacing_matches_the_smoothing():
    """The cross-section had up to 2.3x the spacing that the beam smoothing assumed (3.6 m vs 2 m here), so the layer
    was smoothed too much across the wind. Near the zenith the beam-scale power removed by the smoothing must match
    the truth: master removes 1.37x, and the fix 0.98x (1.05x with the grid padding). With the padding but the old
    spacing it is 1.12x, which the upper limit must catch (over 20 groups of 4 seeds: 1.024-1.080 fixed, 1.121-1.235
    not; the seeds here, 0-3, give 1.045 and 1.121)."""
    instrument, atmosphere = windless_layer(field_of_view=0.6, elevation=88)
    (process,) = atmosphere.processes.values()
    assert np.diff(process.cross_section[:, 0]).max() <= process.layers.res.iloc[0] + 1e-9

    removed, error = smoothing_statistics(instrument, atmosphere)
    assert 0.85 < removed < 1.10
    assert error < 0.04


def test_grid_is_padded_for_the_beam_smoothing():
    """The grid of a layer extended only 1-2 resolution elements past the sampled points, less than the reach of the
    beam smoothing, so the (reflecting) edges of the grid were smoothed into the beam. For a single detector near
    the zenith the beam-scale power removed by the smoothing must match the truth (0.80x without the padding; over
    20 groups of 6 seeds: 0.967-1.049 with it, 0.709-0.869 without)."""
    instrument, atmosphere = windless_layer(field_of_view=0.0, elevation=88, n=1)
    (process,) = atmosphere.processes.values()
    layer = process.layers.iloc[0]
    margin = 3 * instrument.dets.physical_fwhm(layer.z).m.max() / 2.355
    assert np.ptp(process.cross_section[:, 0]) >= 2 * margin
    assert np.ptp(process.extrusion) >= 2 * margin - layer.res

    removed, error = smoothing_statistics(instrument, atmosphere, n_seeds=6)
    assert 0.9 < removed < 1.12
    assert error < 0.035
