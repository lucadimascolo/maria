from __future__ import annotations

import numpy as np
from maria.map import ProjectionMap

CENTER = (150.0, -30.0)


def test_jy_per_beam_input_map(simulate_map):
    """A Jy/beam input map gives the same TOD as the same sky given in Jy/pixel."""
    data = np.abs(np.random.default_rng(1).standard_normal((64, 64)))
    m_pixel = ProjectionMap(data=data, resolution=2 / 3600, nu=150e9, center=CENTER, units="Jy/pixel", beam=8 / 3600)
    m_beam = m_pixel.to("Jy/beam")
    assert m_beam.u["physical_quantity"] == "spectral_flux_density_per_beam"

    tod_pixel = simulate_map(m_pixel).run(units="K_RJ")[0].signal.compute()
    tod_beam = simulate_map(m_beam).run(units="K_RJ")[0].signal.compute()
    assert np.std(tod_pixel) > 0
    assert np.allclose(tod_beam, tod_pixel, rtol=1e-4, atol=1e-4 * np.abs(tod_pixel).max())


def test_map_beam_varying_along_stokes(simulate_map):
    """Only maps in units per beam need a single beam per channel."""
    beam = np.array([[[10 / 3600, 10 / 3600, 0]], [[20 / 3600, 20 / 3600, 0]]])
    data = 1e-3 * np.ones((2, 1, 32, 32))
    m = ProjectionMap(data=data, stokes="IQ", nu=[150e9], width=0.1, center=CENTER, units="K_RJ", beam=beam)
    assert np.allclose(simulate_map(m).run(units="K_RJ")[0].signal.compute(), 1e-3, rtol=1e-3)
