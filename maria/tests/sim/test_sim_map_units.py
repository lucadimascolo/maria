from __future__ import annotations

import numpy as np
import pytest
from maria.beam import compute_angular_fwhm
from maria.coords import offsets_to_phi_theta
from maria.map import ProjectionMap
from scipy.special import erf, erfc

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


@pytest.mark.parametrize("no_data", ["zero weight"])
def test_map_pixels_without_data_are_no_sky(simulate_map, no_data):
    """Pixels without data (of zero weight) are no sky, like the sky beyond the edges of the map: a map of
    10 mK_RJ on one half and no data on the other is seen at 10 mK_RJ * erfc(d / sqrt(2) sigma) / 2 a distance d
    past the boundary, i.e. the half map convolved with the beam, and at 0 beyond 4 sigma (the smoothing used to fill
    them with the sky of their neighbours, at full brightness out to 4 sigma)."""
    fwhm = compute_angular_fwhm(fwhm_0=10, z=np.inf, nu=150e9)
    sigma, res, n = fwhm / np.sqrt(8 * np.log(2)), fwhm / 20, 161
    # the map is 10 mK_RJ for xi < 0 (columns < 80); the boundary is half a pixel before column 80. The samples are
    # at -2, 0.5, 2, 3 and 5 sigma from it (the smoothing kernel reaches 4 sigma).
    data, weight = 1e-2 * np.ones((n, n)), np.ones((n, n))
    if no_data == "zero weight":
        data[:, 80:], weight[:, 80:] = 5e-3, 0
    m = ProjectionMap(data=data, weight=weight, resolution=np.degrees(res), nu=150e9, center=CENTER, units="K_RJ")
    for column in [63, 84, 97, 105, 122]:
        d = (column - 79.5) * res
        center = np.degrees(offsets_to_phi_theta(np.array([m.xi.rad[column], 0.0]), *np.radians(CENTER)))
        tod = simulate_map(m, center=(float(center[0]), float(center[1])), radius=1e-5, duration=5).run(units="K_RJ")
        # (the beam of the smoothing is truncated at 4 sigma)
        expected = 1e-2 * max(erfc(d / (np.sqrt(2) * sigma)) - erfc(2 * np.sqrt(2)), 0) / 2 / erf(2 * np.sqrt(2))
        assert np.allclose(tod[0].signal.compute(), expected, rtol=1e-2, atol=1e-8)
