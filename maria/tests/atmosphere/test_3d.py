from __future__ import annotations

import numpy as np
import pytest
from maria.atmosphere.process import AutoregressiveProcess
from maria.tests.atmosphere._utils import atmosphere_sim, small_instrument, stare_plan

MAX_SAMPLES = 10000  # a covariance matrix of 1e4 x 1e4 points is 0.8 GB


@pytest.fixture(scope="module")
def sim_3d():
    """A 3d atmosphere at an azimuth where the hull is strongly tilted. Building it must not need huge covariance
    matrices (which would exhaust the memory): each sublayer only needs the part of the hull at its own height."""
    original = AutoregressiveProcess.compute_covariance_matrices

    def compute_covariance_matrices(self):
        assert self.n_sample < MAX_SAMPLES, f"the 3d process has {self.n_sample} sample points"
        return original(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AutoregressiveProcess, "compute_covariance_matrices", compute_covariance_matrices)
        np.random.seed(0)
        return atmosphere_sim(
            small_instrument(),
            stare_plan(az=100, el=60, duration=10, sample_rate=5),
            site="llano_de_chajnantor",
            atmosphere="3d",
        )


def test_3d_atmosphere_can_be_simulated(sim_3d):
    atmosphere = sim_3d.obs_list[0].atmosphere
    np.random.seed(1)
    atmosphere.simulate_pwv(instrument=sim_3d.instrument)
    assert np.isfinite(atmosphere.zenith_scaled_pwv).all()

    # the simulated fluctuations scale with the amplitude correction of the 3d process (A10)
    fluctuations = atmosphere.zenith_scaled_pwv - atmosphere.weather.pwv.mm
    process = atmosphere.processes[0]
    correction = process.amplitude_correction
    try:
        process.amplitude_correction = 2 * correction
        np.random.seed(1)
        atmosphere.simulate_pwv(instrument=sim_3d.instrument)
    finally:
        process.amplitude_correction = correction
    assert np.allclose(atmosphere.zenith_scaled_pwv - atmosphere.weather.pwv.mm, 2 * fluctuations, rtol=1e-4, atol=1e-5)


def test_3d_process_wind_is_the_water_weighted_layer_wind(sim_3d):
    """All 3d sublayers share one process velocity, which must be a (water-weighted) mean of the layer winds."""
    atmosphere = sim_3d.obs_list[0].atmosphere
    layers, process = atmosphere.layers, atmosphere.processes[0]
    w = layers.total_water.values
    # the process frame is x = north, y = east
    for process_v, layer_v in [(process.vx, layers.wind_north.values), (process.vy, layers.wind_east.values)]:
        assert layer_v.min() <= np.mean(process_v) <= layer_v.max()
        assert np.mean(process_v) == pytest.approx(np.sum(w * layer_v) / w.sum(), rel=1e-6)


def test_3d_pwv_rms(sim_3d):
    """The 3d sublayers are one correlated field. The PWV along a line of sight (the sum over sublayers) must still
    have an rms of pwv_rms_frac * pwv; summing them as if they were independent gives ~8x that here."""
    atmosphere = sim_3d.obs_list[0].atmosphere
    process = atmosphere.processes[0]
    layers = process.layers

    # the grid point of each sublayer closest to the line of sight of the boresight at the first time sample
    translation = -atmosphere.timestep * np.array([process.vx[0], process.vy[0], 0.0])
    los = (layers.h.values[:, None] * atmosphere.boresight.project(z=1)[0] + translation) @ process.transform
    extrusion_index = np.abs(process.extrusion - los[:, :1]).argmin(axis=1)
    cross_section_index = np.abs(
        np.where(process.labels == layers.index.values[:, None], process.cross_section[:, 0], np.inf) - los[:, 1:2]
    ).argmin(axis=1)

    n_realizations = 50
    los_pwv = np.zeros(n_realizations)
    for i in range(n_realizations):
        np.random.seed(100 + i)
        process.run()
        y = np.asarray(process.values)[extrusion_index, cross_section_index]
        los_pwv[i] = process.amplitude_correction * np.sum(layers.pwv_rms.values * y)

    # the rms of N Gaussian draws is known to 1 / sqrt(2N) = 10%, so this is a 3.5 sigma bound
    rms_frac = np.sqrt(np.mean(los_pwv**2)) / atmosphere.weather.pwv.mm
    assert rms_frac == pytest.approx(atmosphere.pwv_rms_frac, rel=0.35)


def test_3d_without_fluctuations():
    """With pwv_rms_frac=0 the 3d sublayers have zero amplitude, which must give a uniform PWV (not a NaN correction)."""
    sim = atmosphere_sim(
        small_instrument(n=4, field_of_view=0.05),
        stare_plan(duration=10),
        atmosphere="3d",
        atmosphere_kwargs={"layers": {"pwv_rms_frac": 0}},
    )
    atmosphere = sim.obs_list[0].atmosphere
    atmosphere.simulate_pwv(instrument=sim.instrument)
    assert all(process.amplitude_correction == 1.0 for process in atmosphere.processes.values())
    assert np.allclose(atmosphere.zenith_scaled_pwv, atmosphere.weather.pwv.mm)
