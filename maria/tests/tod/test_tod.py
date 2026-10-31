from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from maria import Simulation
from maria.coords import Coordinates, unjitted_offsets_to_phi_theta
from maria.instrument import get_instrument
from maria.noise import generate_noise_with_knee
from maria.plan import get_plan
from maria.tod.tod import TOD

plt.close("all")


def test_tod_functions():
    n = 256

    time = 1.75e9 + np.arange(0, 600, 0.1)
    azim = np.radians(45) * np.ones(len(time))
    elev = np.radians(45) * np.ones(len(time))

    offsets = np.radians(1e1 * np.random.standard_normal(size=(n, 1, 2)))

    PT = unjitted_offsets_to_phi_theta(offsets, azim, elev)

    coords = Coordinates(phi=PT[..., 0], theta=PT[..., 1], t=time, frame="az/el")

    noise = generate_noise_with_knee(shape=coords.shape, sample_rate=1 / coords.timestep, knee=0.5)

    tod = TOD(data=dict(noise=noise), coords=coords)

    print(f"{tod.sample_rate =}")
    print(f"{tod.duration =}")
    print(f"{tod.boresight =}")


def test_tod_calibration_with_atmosphere():
    sim = Simulation(
        instrument="MUSTANG-2",
        plans="ten_second_zenith_stare",
        site="green_bank",
        atmosphere="2d",
        noise=True,
    )

    tod = sim.run()[0]

    tod = tod.to("pW")
    tod = tod.to("K_RJ")


def test_tod_preprocessing_with_config():
    sim = Simulation(
        instrument="MUSTANG-2",
        plans="ten_second_zenith_stare",
        site="green_bank",
        noise=True,
    )

    tod = sim.run()[0][::2, :10000]

    pp_config = {
        "window": {"name": "tukey", "kwargs": {"alpha": 0.25}},
        "filter": {"f_lower": 0.5},
        "remove_modes": {"modes_to_remove": 1},
        "remove_spline": {"knot_spacing": 0.5},
    }

    tod.process(config=pp_config)

    tod.twinkle(
        rate=2,
        max_frames=10,
        filename="/tmp/test_twinkle.gif",
    )


def test_tod_preprocessing_with_kwargs():
    sim = Simulation(
        instrument="MUSTANG-2",
        plans="ten_second_zenith_stare",
        site="green_bank",
        noise=True,
    )

    tod = sim.run()[0]

    tod.process(window="tukey")


def test_tod_preprocessing_errors():
    sim = Simulation(
        instrument="MUSTANG-2",
        plans="ten_second_zenith_stare",
        site="green_bank",
        noise=True,
    )

    tod = sim.run()[0]

    try:
        tod.process(f_lower="a")
        assert False
    except TypeError:
        pass


def test_tod_downsample_puts_the_coordinates_at_the_block_centres():
    """The downsampled data are block means, so the pointing must be that of the block centres (master put it at the
    start of each block: a lag of (factor - 1) / 2 samples, 162 arcsec for 1 deg/s at 100 Hz and factor 10)."""
    time = 1.75e9 + np.arange(6000) / 100

    def track(t):  # 1 deg/s in azimuth, for 3 detectors
        return np.radians(30 + (t[None] - time[0]) + np.arange(3)[:, None])

    coords = Coordinates(phi=track(time), theta=np.radians(np.full((3, time.size), 50.0)), t=time, frame="az/el")
    tod = TOD(data={"noise": np.zeros((3, time.size))}, coords=coords)
    ds = tod.downsample(factor=10)

    assert ds.shape == (3, 600)
    assert np.allclose(ds.coords.t, time.reshape(600, 10).mean(axis=-1), rtol=0, atol=1e-6)
    assert np.degrees(np.abs(ds.coords.az - track(ds.coords.t))).max() * 3600 < 1e-3
