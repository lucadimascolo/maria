from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.instrument import Band
from maria.map import ProjectionMap

CENTER = (150.0, -30.0)


def run(stokes, roll, seed=0):
    """The TOD (K_RJ) of 8 polarized detectors staring at a uniform map with 1 mK_RJ in one Stokes parameter."""
    band = Band(center=150e9, width=10e9, NET_RJ=1e-5)
    np.random.seed(seed)  # for the detector angles and the gain errors
    instrument = maria.get_instrument(
        array={"n": 8, "field_of_view": 0.05, "bands": [band], "primary_size": 10, "polarized": True}
    )
    data = np.zeros((3, 100, 100))
    data["IQU".index(stokes)] = 1e-3
    m = ProjectionMap(data=data, stokes="IQU", width=2.0, nu=150e9, center=CENTER, units="K_RJ")
    plan = maria.Plan.generate(
        start_time="2025-01-01T07:00:00",
        scan_type="daisy",
        scan_parameters={"radius": 0.05, "speed": 0.02},
        duration=5,
        sample_rate=10,
        scan_center=CENTER,
        frame="ra/dec",
        roll=roll,
    )
    sim = maria.Simulation(
        instrument=instrument, plans=plan, site="cerro_toco", map=m, noise=False, atmosphere=None, progress_bars=False
    )
    tod = sim.run(units="K_RJ")[0]
    return tod.signal.compute().mean(axis=-1), np.asarray(instrument.dets.gamma), np.asarray(tod.dets.gamma)


@pytest.mark.parametrize("roll", [0.0, 30.0])
def test_roll_rotates_the_polarization_angles(roll):
    """A detector of angle gamma on a plan with roll r sees (I + Q cos 2(gamma + r) + U sin 2(gamma + r)) / 2."""
    intensity, gamma, tod_gamma = run("I", roll)
    Q, _, _ = run("Q", roll)
    U, _, _ = run("U", roll)
    angle = gamma + np.radians(roll)
    assert np.allclose(Q / intensity, np.cos(2 * angle), atol=1e-3)
    assert np.allclose(U / intensity, np.sin(2 * angle), atol=1e-3)

    # the TOD carries the angles it was simulated with, so that mappers demodulate it with them
    assert np.allclose(tod_gamma, angle)
