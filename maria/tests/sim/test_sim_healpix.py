from __future__ import annotations

import healpy as hp
import numpy as np
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
