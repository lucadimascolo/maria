from __future__ import annotations

import maria
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from maria.coords import offsets_to_phi_theta
from maria.map import ProjectionMap


def maria_pixel_centers(m):
    """The sky position (in degrees) of each pixel of a ProjectionMap, as maria samples it."""
    offsets = np.stack(np.meshgrid(m.xi.rad, m.eta.rad), axis=-1)
    phi_theta = offsets_to_phi_theta(offsets, m.center[0].rad, m.center[1].rad)
    return np.degrees(np.asarray(phi_theta))


def angular_distance_deg(phi1, theta1, phi2, theta2):
    phi1, theta1, phi2, theta2 = map(np.radians, (phi1, theta1, phi2, theta2))
    h = np.sin((theta2 - theta1) / 2) ** 2 + np.cos(theta1) * np.cos(theta2) * np.sin((phi2 - phi1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(h)))


@pytest.mark.parametrize(
    "shape, res, center",
    [
        ((64, 64), 36 / 3600, (150.0, -30.0)),
        ((65, 65), 36 / 3600, (150.0, -30.0)),
        ((33, 64), 36 / 3600, (150.0, -30.0)),
        # maria's projection is zenithal equidistant (ARC), which matters for large maps
        ((201, 201), 180 / 3600, (150.0, -30.0)),
        ((201, 201), 180 / 3600, (150.0, 80.0)),
    ],
)
def test_fits_export_wcs(shape, res, center):
    """The WCS that maria writes puts every pixel where maria samples it."""
    m = ProjectionMap(data=np.zeros(shape), resolution=res, center=center, units="K_RJ")
    header = m.header()
    assert header["CTYPE1"].endswith("ARC") and header["CTYPE2"].endswith("ARC")
    j, i = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    phi, theta = WCS(header).wcs_pix2world(i, j, 0)
    true = maria_pixel_centers(m)
    assert angular_distance_deg(phi, theta, true[..., 0], true[..., 1]).max() < 0.01 * res


def write_image(path, data, header):
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path, overwrite=True)


def reference_pixel_wcs(crpix, dec, ctype=("RA---ARC", "DEC--ARC")):
    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crval = [150.0, dec]
    w.wcs.cdelt = [-0.01, 0.01]
    w.wcs.crpix = list(crpix)
    return w


@pytest.mark.parametrize("dec", [2.0, -30.0, -60.0])
@pytest.mark.parametrize("crpix", [(1.0, 1.0), (32.5, 32.5), (33.0, 33.0), (80.0, -20.0)])
def test_fits_import_reference_pixel(tmp_path, crpix, dec):
    """maria puts every pixel of an imported map where its WCS does, wherever CRPIX is."""
    ny, nx = 64, 64
    w = reference_pixel_wcs(crpix, dec)
    path = str(tmp_path / "image.fits")
    write_image(path, np.zeros((ny, nx)), w.to_header())

    m = maria.map.load(path, units="K_RJ")
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    wphi, wtheta = w.wcs_pix2world(i, j, 0)
    true = maria_pixel_centers(m)
    assert angular_distance_deg(wphi, wtheta, true[..., 0], true[..., 1]).max() < 0.01 * 0.01


def test_fits_import_reference_pixel_with_center_or_size(tmp_path):
    """A center or a resolution passed to load applies to the middle of the map, wherever CRPIX is."""
    ny, nx = 64, 64
    w = reference_pixel_wcs((80.0, -20.0), -30.0)
    path = str(tmp_path / "image.fits")
    write_image(path, np.zeros((ny, nx)), w.to_header())
    middle = w.wcs_pix2world([[(nx - 1) / 2, (ny - 1) / 2]], 0)[0]

    m = maria.map.load(path, units="K_RJ", center=(10.0, 20.0))
    assert np.allclose([m.center[0].deg, m.center[1].deg], [10.0, 20.0])
    assert np.allclose(m.xi.deg, 0.01 * np.linspace(-31.5, 31.5, nx))
    assert np.allclose(m.eta.deg, 0.01 * np.linspace(-31.5, 31.5, ny))

    m = maria.map.load(path, units="K_RJ", resolution=0.02)
    assert angular_distance_deg(m.center[0].deg, m.center[1].deg, *middle) < 1e-3 * 0.01
    assert np.allclose(m.xi.deg, 0.02 * np.linspace(-31.5, 31.5, nx))


def test_fits_import_size_with_a_degenerate_frequency_axis(tmp_path):
    """A size passed to load works for a cube whose frequency axis has CDELT3 = 0, which makes the full WCS singular."""
    ny, nx = 64, 64
    w = reference_pixel_wcs((80.0, -20.0), -30.0)
    middle = w.wcs_pix2world([[(nx - 1) / 2, (ny - 1) / 2]], 0)[0]
    header = w.to_header()
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = "NU", 1.5e11, 0.0, 0.0
    path = str(tmp_path / "cube.fits")
    write_image(path, np.zeros((1, ny, nx)), header)
    m = maria.map.load(path, units="K_RJ", width=1.0)
    assert angular_distance_deg(m.center[0].deg, m.center[1].deg, *middle) < 1e-3 * 0.01


@pytest.mark.parametrize("projection", ["TAN", "SIN"])
def test_fits_import_cutout_far_from_the_tangent_point(tmp_path, projection):
    """A cutout of a TAN (or SIN) map, whose tangent point CRVAL is 3 deg away (as astropy's Cutout2D writes it), is
    centred on its middle pixel: its pixels land where the WCS puts them to a fraction of a pixel (they were 1.6 px
    off when read as ARC about CRVAL)."""
    n, cdelt = 200, 6 / 3600
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    w.wcs.crval, w.wcs.cdelt, w.wcs.crpix = [35.0, -5.0], [-cdelt, cdelt], [100.5, 100.5 - 3.0 / cdelt]
    path = str(tmp_path / "cutout.fits")
    write_image(path, np.zeros((n, n)), w.to_header())
    m = maria.map.load(path, units="K_RJ")
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    wphi, wtheta = w.wcs_pix2world(i, j, 0)
    true = maria_pixel_centers(m)
    error = angular_distance_deg(wphi, wtheta, true[..., 0], true[..., 1]) / cdelt
    assert error[n // 2, n // 2] < 0.01
    assert error.max() < 0.4


def test_fits_import_non_zenithal_projection(tmp_path):
    """A map in a non-zenithal projection (whose CRVAL can be far away) is centred on its middle pixel."""
    ny, nx = 64, 64
    w = reference_pixel_wcs((32.5, 32.5 + 6000.0), 0.0, ctype=("RA---CAR", "DEC--CAR"))
    path = str(tmp_path / "image.fits")
    write_image(path, np.zeros((ny, nx)), w.to_header())
    m = maria.map.load(path, units="K_RJ")
    assert np.allclose([m.center[0].deg, m.center[1].deg], w.wcs_pix2world([[31.5, 31.5]], 0)[0])
    assert np.allclose(m.xi.deg, 0.01 * np.linspace(-31.5, 31.5, nx))


@pytest.mark.parametrize("shape", [(64, 64), (65, 65), (33, 64)])
def test_fits_round_trip(tmp_path, shape):
    rng = np.random.default_rng(0)
    m = ProjectionMap(data=rng.standard_normal(shape), resolution=0.01, center=(150.0, -30.0), units="K_RJ", frame="ra/dec")
    path = str(tmp_path / "map.fits")
    m.to_fits(path)
    m2 = maria.map.load(path)
    assert np.allclose([m2.center[0].deg, m2.center[1].deg], [m.center[0].deg, m.center[1].deg], atol=1e-9)
    assert np.allclose(m2.xi.deg, m.xi.deg) and np.allclose(m2.eta.deg, m.eta.deg)
    assert np.allclose(m2.data.compute(), m.data.compute())
