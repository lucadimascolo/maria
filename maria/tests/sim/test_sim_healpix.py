from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from maria.beam import compute_angular_fwhm
from maria.coords import Coordinates
from maria.map import HEALPixMap

CENTER = (150.0, -30.0)


def disc_map(nside, center, radius_deg, frame="ra/dec"):
    """A HEALPix map that is 1 K_RJ inside a disc and 0 outside."""
    vec = hp.ang2vec(np.radians(90 - center[1]), np.radians(center[0]))
    data = np.zeros(hp.nside2npix(nside))
    data[hp.query_disc(nside, vec, radius=np.radians(radius_deg))] = 1.0
    return HEALPixMap(data=data, units="K_RJ", frame=frame)


def test_healpix_input_map(simulate_map):
    """Simulation(map=<HEALPixMap>) runs, and a uniform 10 mK_RJ sky gives a 10 mK_RJ TOD."""
    m = HEALPixMap(data=1e-2 * np.ones(hp.nside2npix(64)), units="K_RJ")
    sim = simulate_map(m)
    assert np.allclose(sim.run(units="K_RJ")[0].signal.compute(), 1e-2, rtol=1e-3)

    # the smoothed map that the simulation keeps follows a change of map, and an edit of the map in place
    sim._initialize_map(HEALPixMap(data=2e-2 * np.ones(hp.nside2npix(64)), units="K_RJ"))
    assert np.allclose(sim.run(units="K_RJ")[0].signal.compute(), 2e-2, rtol=1e-3)
    sim.maps["map"].data[:] = 3e-2
    assert np.allclose(sim.run(units="K_RJ")[0].signal.compute(), 3e-2, rtol=1e-3)

    # also when the data are a numpy array, which is the same object after an edit in place
    sim.maps["map"].data = np.asarray(sim.maps["map"].data).copy()
    assert np.allclose(sim.run(units="K_RJ")[0].signal.compute(), 3e-2, rtol=1e-3)
    sim.maps["map"].data *= 1.5
    assert np.allclose(sim.run(units="K_RJ")[0].signal.compute(), 4.5e-2, rtol=1e-3)


def test_healpix_map_with_nan_pixels(simulate_map):
    """NaN pixels are masked like healpy's UNSEEN: NaN far from the scan gives the same TOD as zero there."""
    nside = 64
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    tods = []
    for fill in [0.0, hp.UNSEEN, np.nan]:
        data = np.where(theta < np.pi / 4, fill, 1e-2)  # north of dec +45 deg
        tods.append(simulate_map(HEALPixMap(data=data, units="K_RJ")).run(units="K_RJ")[0].signal.compute())
    assert np.allclose(tods[1], tods[0]) and np.allclose(tods[2], tods[0])


def test_healpix_input_map_is_beam_smoothed(simulate_map):
    """A 1 K_RJ disc of radius R seen at its centre with a Gaussian beam gives 1 - exp(-R^2 / 2 sigma^2)."""
    nside, primary_size = 512, 0.5
    sigma_deg = np.degrees(compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=150e9)) / np.sqrt(8 * np.log(2))
    m = disc_map(nside, CENTER, radius_deg=0.2)
    tod = simulate_map(m, primary_size=primary_size, radius=0.01).run(units="K_RJ")[0].signal.compute()

    # the pixelised disc has a slightly different area than the true one; use its effective radius
    radius_deg = np.degrees(np.sqrt(m.data.sum().compute() * hp.nside2pixarea(nside) / np.pi))
    expected = 1 - np.exp(-(radius_deg**2) / (2 * sigma_deg**2))
    assert 0.3 < expected < 0.8
    assert np.isclose(tod.mean(), expected, rtol=0.05)


def test_galactic_healpix_map():
    """A galactic HEALPix map keeps its frame, and is sampled at the galactic position of the pointing."""
    l0, b0 = 10.0, 20.0
    m = disc_map(128, (l0, b0), radius_deg=1.0, frame="galactic")
    assert m.frame.name == m.copy().frame.name == "galactic"

    on = SkyCoord(l=l0 * u.deg, b=b0 * u.deg, frame="galactic").icrs
    off = SkyCoord(ra=l0 * u.deg, dec=b0 * u.deg, frame="icrs")  # the same (lon, lat) read as RA/Dec
    ra = np.radians([on.ra.deg, off.ra.deg])[:, None]
    dec = np.radians([on.dec.deg, off.dec.deg])[:, None]
    coords = Coordinates(phi=ra, theta=dec, t=1.7e9 * np.ones(1), frame="ra/dec")

    _, pixels, _, _, _ = m._pointing_matrix_ingredients(coords=coords)
    assert np.allclose(m.data.compute()[np.asarray(pixels).ravel()], [1.0, 0.0])


def polarized_sky(nside, lmax, seed=0):
    """A band-limited IQU sky in ra/dec, and the same sky expressed in galactic coordinates."""
    ell = np.arange(lmax + 1)
    cl = np.zeros((4, lmax + 1))
    cl[:3, 2:] = 1e-6 / ell[2:] ** 2  # TT, EE, BB
    np.random.seed(seed)
    iqu_radec = hp.synfast(cl, nside, lmax=lmax, new=True)
    iqu_galactic = hp.Rotator(coord=["C", "G"]).rotate_map_alms(iqu_radec, lmax=lmax)
    return iqu_radec, iqu_galactic


def test_galactic_polarized_map_is_rotated():
    """Q and U of a galactic map refer to galactic north; smoothing it to ra/dec refers them to celestial north."""
    nside, lmax, fwhm = 64, 64, 2.0
    iqu_radec, iqu_galactic = polarized_sky(nside, lmax)
    m = HEALPixMap(data=iqu_galactic, stokes="IQU", units="K_RJ", frame="galactic")
    smoothed = m._smooth(fwhm=fwhm, pol=True, frame="ra/dec")
    assert smoothed.frame.name == "ra/dec"
    expected = hp.smoothing(iqu_radec, fwhm=np.radians(fwhm), pol=True, lmax=lmax)
    got = np.asarray(smoothed.data).reshape(3, -1)
    for stokes, g, e in zip("IQU", got, expected):
        assert np.std(g - e) < 1e-2 * np.std(e), stokes


def test_galactic_polarized_map_in_the_simulation(simulate_map):
    """The same polarized sky given in galactic or in ra/dec coordinates gives the same TOD."""
    import maria
    from maria.instrument import Band

    nside, lmax = 64, 64
    iqu_radec, iqu_galactic = polarized_sky(nside, lmax)
    band = Band(center=150e9, width=10e9, NET_RJ=1e-5)
    instrument = maria.get_instrument(
        array={"n": 7, "field_of_view": 0.5, "bands": [band], "primary_size": 1, "polarized": True}
    )
    tods = [
        simulate_map(HEALPixMap(data=iqu, stokes="IQU", units="K_RJ", frame=frame), instrument=instrument, radius=1.0)
        .run(units="K_RJ")[0]
        .signal.compute()
        for iqu, frame in [(iqu_radec, "ra/dec"), (iqu_galactic, "galactic")]
    ]
    assert np.std(tods[1] - tods[0]) < 2e-2 * np.std(tods[0])


def test_galactic_polarized_projection_map(simulate_map):
    """A galactic ProjectionMap gives the same polarized TOD as the same sky as a galactic HEALPixMap."""
    import maria
    from maria.instrument import Band
    from maria.map import ProjectionMap

    band = Band(center=150e9, width=10e9, NET_RJ=1e-5)
    array = {"n": 4, "field_of_view": 0.02, "bands": [band], "primary_size": 6, "polarized": True}
    instrument = maria.get_instrument(array=array)
    galactic = SkyCoord(ra=CENTER[0] * u.deg, dec=CENTER[1] * u.deg).galactic
    l0, b0 = galactic.l.deg, galactic.b.deg

    # Q = 1 mK_RJ (referred to galactic north) around the scan, as a flat map and as a broad HEALPix bump
    data = np.zeros((3, 64, 64))
    data[1] = 1e-3
    flat = ProjectionMap(data=data, stokes="IQU", resolution=0.01, center=(l0, b0), units="K_RJ", frame="galactic")
    nside = 128
    distance = hp.rotator.angdist(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True), (l0, b0), lonlat=True)
    bump = 1e-3 * np.exp(-0.5 * (distance / np.radians(10)) ** 2)
    sphere = HEALPixMap(data=np.stack([0 * bump, bump, 0 * bump]), stokes="IQU", units="K_RJ", frame="galactic")

    flat_tod, sphere_tod = (
        simulate_map(m, instrument=instrument, radius=0.01, duration=5).run(units="K_RJ")[0].signal.compute().mean(axis=1)
        for m in (flat, sphere)
    )
    assert np.abs(sphere_tod).max() > 5e-4
    assert np.allclose(flat_tod, sphere_tod, atol=1e-5)


def test_galactic_polarized_map_is_rotated_once(simulate_map, monkeypatch):
    """Smoothing and rotating commute, so the harmonic coefficients are rotated once, whatever the number of bands."""
    import maria
    from maria.instrument import Band

    calls = []
    rotate_alm = hp.Rotator.rotate_alm

    def counting_rotate_alm(self, *args, **kwargs):
        calls.append(1)
        return rotate_alm(self, *args, **kwargs)

    monkeypatch.setattr(hp.Rotator, "rotate_alm", counting_rotate_alm)

    _, iqu_galactic = polarized_sky(nside=32, lmax=32)
    bands = [Band(center=nu, width=10e9, NET_RJ=1e-5) for nu in (90e9, 150e9)]
    array = {"n": 2, "field_of_view": 0.5, "bands": bands, "primary_size": 1, "polarized": True}
    instrument = maria.get_instrument(array=array)
    m = HEALPixMap(data=iqu_galactic, stokes="IQU", units="K_RJ", frame="galactic")
    calls.clear()
    simulate_map(m, instrument=instrument, duration=2).run()
    assert len(calls) == 1


def test_smooth_frame_rotates_the_weight():
    """The weight is rotated with the map: a weight on the northern galactic hemisphere stays there."""
    nside = 32
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    weight = (theta < np.pi / 2).astype(float)  # galactic latitude > 0
    m = HEALPixMap(data=np.ones_like(weight), weight=weight, units="K_RJ", frame="galactic")
    smoothed = m._smooth(fwhm=1.0, frame="ra/dec")

    b = np.pi / 2 - hp.Rotator(coord=["C", "G"])(theta, phi)[0]  # galactic latitude of each ra/dec pixel
    w, edge = np.asarray(smoothed.weight), 2 * hp.nside2resol(nside)
    assert (w[b > edge] == 1).all() and (w[b < -edge] == 0).all()


def test_smooth_frame_aliases_and_pol():
    """_smooth(frame=...) takes any name of a frame, and does not rotate Q and U as scalars."""
    _, iqu_galactic = polarized_sky(nside=16, lmax=16)
    m = HEALPixMap(data=iqu_galactic[0], units="K_RJ", frame="galactic")
    assert m._smooth(fwhm=1.0, frame="icrs").frame.name == "ra/dec"
    assert m._smooth(fwhm=1.0, frame="glon/glat").frame.name == "galactic"

    m = HEALPixMap(data=iqu_galactic, stokes="IQU", units="K_RJ", frame="galactic")
    with pytest.raises(ValueError):
        m._smooth(fwhm=1.0, frame="ra/dec")
