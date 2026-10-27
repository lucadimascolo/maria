"""
Truth tests for maria's pointing: az/el <-> ra/dec and galactic against astropy.

maria's az/el are geometric (no refraction), so the reference is astropy's AltAz frame with pressure = 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from astropy.time import Time
from maria.coords import Coordinates, offsets_to_phi_theta
from maria.site import get_site

T0 = 1.7e9  # 2023-11-14 22:13:20 UTC
ARCSEC = np.degrees(1) * 3600


def separation_arcsec(phi1, theta1, phi2, theta2):
    return angular_separation(*(np.asarray(a, dtype=np.float64) for a in (phi1, theta1, phi2, theta2))) * ARCSEC


def altaz_frame(t, earth_location):
    return AltAz(obstime=Time(np.ravel(t), format="unix"), location=earth_location, pressure=0 * u.hPa)


def true_sky(az, el, t, earth_location, frame="icrs"):
    """The true (lon, lat) in an astropy frame of geometric (az, el)."""
    az, el, t = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (az, el, t)))
    el = np.minimum(el, np.pi / 2)  # float32 elevations can exceed 90 deg by 2.5e-6 deg
    c = SkyCoord(az=az.ravel() * u.rad, alt=el.ravel() * u.rad, frame=altaz_frame(t, earth_location))
    c = c.transform_to(frame).spherical
    return c.lon.rad.reshape(az.shape), c.lat.rad.reshape(az.shape)


def true_azel(ra, dec, t, earth_location):
    ra, dec, t = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (ra, dec, t)))
    c = SkyCoord(ra=ra.ravel() * u.rad, dec=dec.ravel() * u.rad, frame="icrs").transform_to(altaz_frame(t, earth_location))
    return c.az.rad.reshape(ra.shape), c.alt.rad.reshape(ra.shape)


def grid_offsets(n=5, spacing_deg=1.0):
    g = np.radians(spacing_deg * (np.arange(n) - (n - 1) / 2))
    return np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)


def boresight(t, phi, theta, site="cerro_toco", frame="az/el"):
    """A boresight at (phi, theta) in degrees, constant or varying with t."""
    phi, theta = (np.radians(a) * np.ones(np.shape(t)) for a in (phi, theta))
    return Coordinates(phi=phi, theta=theta, t=t, earth_location=get_site(site).earth_location, frame=frame)


@pytest.mark.parametrize(
    "site, el",
    [("cerro_toco", 30.0), ("cerro_toco", 60.0), ("cerro_toco", 85.0), ("cerro_toco", 90.0), ("amundsen_scott", 45.0)],
)
def test_azel_to_sky_matches_astropy(site, el):
    """P1: the frame transform was fitted on fiducials spread over the whole sphere (radii in radians), ~5-26".
    The fit is accurate to ~0.1" near the boresight, and the error grows with the square of the offset (~0.3" at 10 deg).
    """
    t = T0 + np.arange(0, 60, 1.0)
    psi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.radians(10.0) * np.c_[np.cos(psi), np.sin(psi)]
    coords = boresight(t, 30.0, el, site=site).broadcast(np.r_[grid_offsets(), ring])

    for (lon, lat), frame in [(("ra", "dec"), "icrs"), (("l", "b"), "galactic")]:
        true_lon, true_lat = true_sky(coords.az, coords.el, t, coords.earth_location, frame=frame)
        err = separation_arcsec(getattr(coords, lon), getattr(coords, lat), true_lon, true_lat)
        assert err.max() < 0.5, f"{frame}: max error {err.max():.3f} arcsec"


def test_radec_to_azel_matches_astropy():
    """P1, reverse direction: an ra/dec boresight converted to az/el."""
    t = T0 + np.arange(0, 60, 1.0)
    ra0, dec0 = np.degrees(true_sky(np.radians(30.0), np.radians(50.0), T0, get_site("cerro_toco").earth_location))
    coords = boresight(t, ra0, dec0, frame="ra/dec")

    az, el = true_azel(coords.ra, coords.dec, t, coords.earth_location)
    err = separation_arcsec(coords.az, coords.el, az, el)

    assert err.max() < 0.5, f"max error {err.max():.3f} arcsec"


def test_long_span_matches_astropy():
    """P2: transform nodes every 600 s with a quadratic interpolation gave ~2.6" at a 1700 s span."""
    t = T0 + np.arange(0, 1700, 1.0)
    coords = boresight(t, 120.0, 50.0)

    ra, dec = true_sky(coords.az, coords.el, t, coords.earth_location)
    err = separation_arcsec(coords.ra, coords.dec, ra, dec)

    assert err.max() < 0.2, f"max error {err.max():.3f} arcsec"


def test_transform_does_not_depend_on_time_span():
    """P2: the transform nodes were spread over the span of the coordinates, so the same sample got a different
    ra/dec when the coordinates covered a different time range (e.g. a time chunk)."""
    t = T0 + 13.7 + np.arange(0, 1800, 0.5)
    az = 120 + 5 * np.sin(2 * np.pi * (t - T0) / 37.0)

    full = boresight(t, az, 50.0).broadcast(grid_offsets(n=3))
    n = len(t) // 4
    for s in [slice(0, n), slice(n, 2 * n)]:
        part = boresight(t[s], az[s], 50.0).broadcast(grid_offsets(n=3))
        err = separation_arcsec(part.ra, part.dec, full.ra[:, s], full.dec[:, s])
        assert err.max() < 0.1, f"max difference {err.max():.3f} arcsec"


def test_wide_scan_matches_astropy():
    """P2: a back-and-forth scan of +-20 deg in az at 2 deg/s. The transform at each node is fitted around the
    boresight at that node, so the accuracy and the chunk independence degrade when the boresight moves far within the
    +-2 nodes (~0.75" and ~0.25" here, against ~0.1" and ~0.03" for a +-5 deg throw)."""
    t = T0 + 13.7 + np.arange(0, 1200, 0.5)
    az = 120 + 20 * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * 2.0 * (t - T0) / 80.0))

    full = boresight(t, az, 45.0).broadcast(grid_offsets(n=3))
    ra, dec = true_sky(full.az, full.el, t, full.earth_location)
    err = separation_arcsec(full.ra, full.dec, ra, dec)
    assert err.max() < 1.0, f"max error {err.max():.3f} arcsec"

    n = len(t) // 4
    for s in [slice(0, n), slice(2 * n + 17, 3 * n + 101)]:
        part = boresight(t[s], az[s], 45.0).broadcast(grid_offsets(n=3))
        err = separation_arcsec(part.ra, part.dec, full.ra[:, s], full.dec[:, s])
        assert err.max() < 0.5, f"max difference {err.max():.3f} arcsec"
