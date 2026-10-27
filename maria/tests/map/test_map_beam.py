from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
from maria.map import HEALPixMap, ProjectionMap

FWHM_TO_SIGMA = 1 / np.sqrt(8 * np.log(2))


def gaussian_source_map(fwhm_deg: float, flux_jy: float = 1.0, n: int = 257):
    """A Gaussian source of total flux `flux_jy`, as a Jy/pixel map whose beam is the source FWHM."""
    resolution_deg = fwhm_deg / 10
    x = resolution_deg * (np.arange(n) - (n - 1) / 2)
    X, Y = np.meshgrid(x, x)
    profile = np.exp(-0.5 * (X**2 + Y**2) / (fwhm_deg * FWHM_TO_SIGMA) ** 2)
    data = flux_jy * profile / profile.sum()
    return ProjectionMap(data=data[None], nu=[150e9], resolution=resolution_deg, units="Jy/pixel", beam=fwhm_deg)


@pytest.mark.parametrize("major, minor", [(1 / 3600, 1 / 3600), (20 / 3600, 20 / 3600), (0.5, 0.5), (0.02, 0.01)])
def test_beam_area_is_gaussian_solid_angle(major, minor):
    m = ProjectionMap(data=np.zeros((1, 8, 8)), nu=[150e9], resolution=1e-3, units="K_RJ", beam=(major, minor, 30.0))
    sigma_major, sigma_minor = np.radians([major, minor]) * FWHM_TO_SIGMA
    assert np.allclose(m.beam_area.sr, 2 * np.pi * sigma_major * sigma_minor, rtol=1e-6, atol=0)


def test_jy_per_beam_to_rayleigh_jeans_temperature():
    """1 Jy/beam -> T_RJ = c^2 / (2 k nu^2 Omega_beam), with the Gaussian beam solid angle."""
    c, k_B = 299792458.0, 1.380649e-23
    nu, fwhm_deg = 150e9, 30 / 3600
    m = ProjectionMap(data=np.ones((1, 16, 16)), nu=[nu], resolution=fwhm_deg / 5, units="Jy/beam", beam=fwhm_deg)
    omega = 2 * np.pi * (np.radians(fwhm_deg) * FWHM_TO_SIGMA) ** 2
    T_expected = 1e-26 * c**2 / (2 * k_B * nu**2 * omega)
    assert np.allclose(m.to("K_RJ").data.compute(), T_expected, rtol=1e-4)
    # and the Jy/beam and Jy/pixel paths agree
    assert np.allclose(m.to("Jy/pixel").to("K_RJ").data.compute(), T_expected, rtol=1e-4)


def test_smooth_updates_beam():
    m = ProjectionMap(data=np.zeros((1, 32, 32)), nu=[150e9], resolution=5e-3, units="K_RJ", beam=(0.02, 0.01, 30.0))
    assert np.allclose(m.smooth(fwhm=0.05).beam.deg.reshape(-1, 3)[0], [np.hypot(0.02, 0.05), np.hypot(0.01, 0.05), 30.0])

    h = HEALPixMap(data=np.zeros((1, hp.nside2npix(16))), nu=[150e9], units="K_RJ")
    assert np.allclose(h.smooth(fwhm=5.0).beam.deg.reshape(-1, 3)[0], [5.0, 5.0, 0.0])


def test_smooth_jy_per_beam_point_source():
    """A point source of flux S observed with a Gaussian beam peaks at S Jy/beam, also after smoothing, and
    conversions stay consistent."""
    flux = 3.0
    m = gaussian_source_map(fwhm_deg=20 / 3600, flux_jy=flux).to("Jy/beam")
    assert np.isclose(float(m.data.max().compute()), flux, rtol=1e-2)
    s = m.smooth(fwhm=30 / 3600)
    assert np.isclose(float(s.data.max().compute()), flux, rtol=1e-2)
    # the total flux is unchanged
    assert np.isclose(float(s.to("Jy/pixel").data.sum().compute()), flux, rtol=1e-2)
    # and an intensive map smoothed the same way gives the same brightness
    T = m.to("K_RJ").smooth(fwhm=30 / 3600).data.compute()
    assert np.allclose(s.to("K_RJ").data.compute(), T, rtol=1e-4, atol=1e-4 * T.max())
