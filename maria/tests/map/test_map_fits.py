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


@pytest.mark.parametrize(
    "eta, xi",
    [(slice(10, 50), slice(100, 190)), (slice(None, None, 2), slice(1, None, 3)), (slice(None, None, -1), slice(0, 60))],
)
def test_fits_export_wcs_of_sliced_map(eta, xi):
    """A slice of a map has its centre (where the offsets are zero) away from its middle pixel; the WCS follows it."""
    m = ProjectionMap(data=np.zeros((200, 200)), resolution=0.05, center=(150.0, -30.0), units="K_RJ")[..., eta, xi]
    shape = (m.dims["eta"], m.dims["xi"])
    j, i = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    phi, theta = WCS(m.header()).wcs_pix2world(i, j, 0)
    true = maria_pixel_centers(m)
    assert angular_distance_deg(phi, theta, true[..., 0], true[..., 1]).max() < 1e-3 * 0.05


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


@pytest.mark.parametrize("center", [(150.0, -30.0), (150.0, -60.0), (150.0, 2.0)])
def test_fits_round_trip_of_sliced_map(tmp_path, center):
    """A sliced map, whose centre is not in its middle, keeps every pixel where it was through a FITS file."""
    rng = np.random.default_rng(0)
    m = ProjectionMap(data=rng.random((200, 200)), resolution=0.05, center=center, units="K_RJ")[..., 10:50, 100:190]
    path = str(tmp_path / "sliced.fits")
    m.to_fits(path)
    m2 = maria.map.load(path)
    p1, p2 = maria_pixel_centers(m), maria_pixel_centers(m2)
    assert angular_distance_deg(p1[..., 0], p1[..., 1], p2[..., 0], p2[..., 1]).max() < 1e-3 * 0.05
    assert np.allclose(m2.data.compute(), m.data.compute())


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


def test_fits_spectral_and_stokes_axes(tmp_path):
    """Other tools read maria's frequency and Stokes axes correctly, and maria reads standard cubes."""
    nu = [90e9, 150e9, 210e9]
    m = ProjectionMap(data=np.zeros((3, 3, 8, 8)), stokes="IQU", nu=nu, resolution=0.01, units="K_RJ")
    wcs = WCS(m.header())
    # FITS axes: (xi, eta, nu, stokes)
    world = wcs.wcs_pix2world([[0, 0, k, k] for k in range(3)], 0)
    assert np.allclose(world[:, 2], nu)
    assert np.allclose(world[:, 3], [1, 2, 3])  # FITS Stokes codes I, Q, U

    # a standard cube with 1-based reference pixels
    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---ARC", "DEC--ARC", "FREQ", "STOKES"]
    w.wcs.crval = [150.0, -30.0, 90e9, 2.0]  # the first Stokes plane is Q
    w.wcs.cdelt = [-0.01, 0.01, 60e9, 1.0]
    w.wcs.crpix = [4.5, 4.5, 1.0, 1.0]
    w.wcs.cunit = ["deg", "deg", "Hz", ""]
    path = str(tmp_path / "cube.fits")
    write_image(path, np.zeros((3, 3, 8, 8)), w.to_header())
    m2 = maria.map.load(path, units="K_RJ")
    assert np.allclose(m2.nu.Hz, nu)
    assert list(m2.stokes) == ["Q", "U", "V"]


@pytest.mark.parametrize("keyword", ["RESTFRQ", "RESTFREQ", "FREQ"])
def test_fits_frequency_keyword_does_not_override_the_axis(tmp_path, keyword):
    """A cube with a frequency axis keeps it when its header also has a (rest) frequency keyword."""
    w = WCS(naxis=3)
    w.wcs.ctype = ["RA---ARC", "DEC--ARC", "FREQ"]
    w.wcs.crval = [150.0, -30.0, 90e9]
    w.wcs.cdelt = [-0.01, 0.01, 10e9]
    w.wcs.crpix = [4.5, 4.5, 1.0]
    w.wcs.cunit = ["deg", "deg", "Hz"]
    header = w.to_header()
    header[keyword] = 115.27e9
    path = str(tmp_path / "cube.fits")
    write_image(path, np.zeros((3, 8, 8)), header)
    assert np.allclose(maria.map.load(path, units="K_RJ").nu.Hz, [90e9, 100e9, 110e9])

    # a single-channel frequency axis also takes precedence over the keyword
    write_image(path, np.zeros((1, 8, 8)), header)
    assert np.allclose(maria.map.load(path, units="K_RJ").nu.Hz, [90e9])

    # without a frequency axis, the keyword gives the frequency of the map
    header = w.celestial.to_header()
    header[keyword] = 115.27e9
    write_image(path, np.zeros((8, 8)), header)
    assert np.allclose(maria.map.load(path, units="K_RJ").nu.Hz, [115.27e9])


def test_fits_single_channel_wcs(tmp_path):
    """A single-channel map writes a non-singular WCS."""
    m = ProjectionMap(data=np.zeros((1, 8, 8)), nu=[150e9], resolution=0.01, units="K_RJ")
    wcs = WCS(m.header())
    assert np.isclose(wcs.wcs_pix2world([[0, 0, 0]], 0)[0, 2], 150e9)

    path = str(tmp_path / "map.fits")
    m.to_fits(path)
    assert np.allclose(maria.map.load(path).nu.Hz, [150e9])


def test_fits_round_trip_cube(tmp_path):
    rng = np.random.default_rng(1)
    shape = (3, 2, 4, 8, 9)
    m = ProjectionMap(
        data=rng.standard_normal(shape),
        weight=rng.uniform(1, 2, size=shape),
        stokes="IQU",
        nu=[90e9, 150e9],
        t=1.7e9 + 60.0 * np.arange(4),
        resolution=0.01,
        center=(10.0, 20.0),
        units="K_RJ",
        beam=(0.02, 0.01, 30.0),
    )
    header = m.header()
    # the standard beam keywords, which CASA, radio-beam etc. read
    assert np.isclose(header["BMAJ"], 0.02) and np.isclose(header["BMIN"], 0.01) and np.isclose(header["BPA"], 30.0)

    path = str(tmp_path / "cube.fits")
    m.to_fits(path)
    m2 = maria.map.load(path)
    assert list(m2.stokes) == list(m.stokes)
    assert np.allclose(m2.nu.Hz, m.nu.Hz)
    assert np.allclose(m2.t.seconds, m.t.seconds)
    assert np.allclose(m2.data.compute(), m.data.compute())
    assert np.allclose(m2.weight.compute(), m.weight.compute())
    assert np.allclose(m2.beam.deg.reshape(-1, 3)[0], [0.02, 0.01, 30.0])


@pytest.mark.parametrize(
    "dim, ctype, crval, cdelt, n, expected",
    [
        ("stokes", "STOKES", 0, 0.0, 1, ["I"]),
        ("stokes", "STOKES", 0, 1.0, 3, ["I", "Q", "U"]),
        ("nu", "NU", 90e9, 60e9, 3, [90e9, 150e9, 210e9]),
        ("t", "TIME", 1.7e9, 10.0, 3, [1.7e9, 1.7e9 + 10, 1.7e9 + 20]),
    ],
)
def test_fits_read_old_maria_axes(tmp_path, dim, ctype, crval, cdelt, n, expected):
    """Files written by older maria (CRPIX = 0 and CRVAL at the first pixel, 0-based Stokes) keep their axes."""
    header = fits.Header()
    for key, value in {"CTYPE1": "RA---SIN", "CRVAL1": 150.0, "CRPIX1": 4, "CDELT1": -0.01}.items():
        header[key] = value
    for key, value in {"CTYPE2": "DEC--SIN", "CRVAL2": -30.0, "CRPIX2": 4, "CDELT2": 0.01}.items():
        header[key] = value
    for key, value in {"CTYPE3": ctype, "CRVAL3": crval, "CRPIX3": 0, "CDELT3": cdelt}.items():
        header[key] = value
    header["BUNIT"] = "K_RJ"
    path = str(tmp_path / "old.fits")
    write_image(path, np.zeros((n, 8, 8)), header)

    values = getattr(maria.map.load(path), dim)
    if dim == "stokes":
        assert list(values) == expected
    else:
        assert np.allclose(values.base_units_value, expected, rtol=0, atol=1e-3 * cdelt)
