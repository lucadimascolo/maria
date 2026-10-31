"""
Truth tests for maria's pointing: az/el <-> ra/dec and galactic against astropy.

maria's az/el are geometric (no refraction), so the reference is astropy's AltAz frame with pressure = 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation, offset_by
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


@pytest.mark.parametrize("el", [60.0, 89.0, 89.9, 90.0])
def test_offsets_near_zenith(el):
    """P3: latitudes from a float32 arcsin are wrong by up to 71" within ~0.5 deg of the zenith or pole."""
    offsets = grid_offsets(n=21, spacing_deg=0.1)
    caz, cel = np.radians(30.0), np.radians(el)

    phi_theta = np.asarray(offsets_to_phi_theta(offsets, caz, cel), dtype=np.float64)

    # maria's offsets: x = -r sin(pa), y = r cos(pa), with pa the position angle (east of north)
    true_phi, true_theta = offset_by(
        lon=caz * u.rad,
        lat=cel * u.rad,
        posang=np.arctan2(-offsets[:, 0], offsets[:, 1]) * u.rad,
        distance=np.hypot(offsets[:, 0], offsets[:, 1]) * u.rad,
    )
    err = separation_arcsec(phi_theta[:, 0], phi_theta[:, 1], true_phi.rad, true_theta.rad)

    assert err.max() < 0.1, f"max error {err.max():.3f} arcsec"
    assert (phi_theta[:, 1] <= np.float32(np.pi / 2)).all()


@pytest.mark.parametrize("el", [40.0, 70.0])
def test_transform_does_not_depend_on_detectors(el):
    """P4: the transform was fitted around the centre of whichever detectors are present, so the same detector got a
    different ra/dec in a sub-array than in the full array (0.9-1.2" for sub-arrays 10 deg apart)."""
    t = T0 + np.arange(0, 600, 5.0)
    sub = grid_offsets(n=3, spacing_deg=0.25)
    offsets = np.r_[sub + np.radians([-5.0, 0.0]), sub + np.radians([5.0, 0.0])]

    full = boresight(t, 120.0, el).broadcast(offsets)
    part = boresight(t, 120.0, el).broadcast(offsets[: len(sub)])

    err = separation_arcsec(part.ra, part.dec, full.ra[: len(sub)], full.dec[: len(sub)])
    assert err.max() < 1e-3, f"max difference {err.max():.4f} arcsec"


@pytest.mark.parametrize("keys", [[slice(6, None)], [slice(2, 9), slice(1, 4)], [(slice(0, 4), slice(10, 50))], [3]])
def test_subset_uses_the_same_transform(keys):
    """P4: a subset (sliced before ra/dec is computed) fitted its own transform around its own centre (0.2-0.8" here)."""
    t = T0 + np.arange(0, 100, 1.0)
    offsets = np.radians(np.c_[np.linspace(-5, 5, 11), np.zeros(11)])

    def subset(coords):
        for key in keys:
            coords = coords[key]
        return coords

    coords = boresight(t, 30.0, 50.0).broadcast(offsets)
    ra, dec = subset(np.asarray(coords.ra)), subset(np.asarray(coords.dec))
    sliced = subset(boresight(t, 30.0, 50.0).broadcast(offsets))

    err = separation_arcsec(sliced.ra, sliced.dec, ra, dec)
    assert err.max() < 1e-3, f"max difference {err.max():.4f} arcsec"


@pytest.mark.parametrize("key", [(slice(None), slice(30, 70)), (slice(None), slice(5, 6))])
def test_time_slice_after_a_transform(key):
    """A time slice of coordinates that already computed one frame must compute the others on its own times."""
    t = T0 + np.arange(0, 100, 1.0)
    offsets = np.radians(np.c_[np.linspace(-1, 1, 5), np.zeros(5)])
    coords = boresight(t, 30.0, 50.0).broadcast(offsets)
    coords.ra  # noqa: B018 (initializes the transforms)
    sliced = coords[key]
    full = boresight(t, 30.0, 50.0).broadcast(offsets)
    assert sliced.l.size == sliced.phi.size  # (a 1-sample slice may drop its time axis, as on master)
    assert np.allclose(sliced.l, full.l[key].reshape(sliced.l.shape), atol=1e-10)
    assert np.allclose(sliced.b, full.b[key].reshape(sliced.b.shape), atol=1e-10)


def test_band_slice_uses_the_same_transform():
    """P-new2: the map and CMB are sampled with obs.coords[band_mask], which fitted its own transform, while the
    mappers use the full obs.coords (1.45" for two single-band arrays 10 deg apart)."""
    import maria
    from maria.instrument import Band
    from maria.plan import Plan
    from maria.sim.observation import Observation

    f090 = Band(center=90e9, width=20e9, NET_RJ=1e-5)
    f150 = Band(center=150e9, width=30e9, NET_RJ=1e-5)
    instrument = maria.get_instrument(
        arrays=[
            dict(field_of_view=0.5, bands=[f090], primary_size=6, focal_plane_offset=[-5.0, 0.0]),
            dict(field_of_view=0.5, bands=[f150], primary_size=6, focal_plane_offset=[5.0, 0.0]),
        ]
    )
    plan = Plan.generate(
        start_time=T0,
        duration=600,
        sample_rate=2,
        scan_type="daisy",
        frame="az/el",
        scan_center=(120.0, 50.0),
        scan_parameters={"radius": 0.5, "speed": 0.1},
        site="cerro_toco",
    )
    obs = Observation(instrument=instrument, plan=plan, site=get_site("cerro_toco"))

    for band_name in ["f090", "f150"]:
        band_mask = np.asarray(obs.instrument.dets.band_name == band_name)
        band_coords = obs.coords[band_mask]  # as in sim/map.py and sim/cmb.py
        for lon, lat in [("ra", "dec"), ("l", "b")]:
            err = separation_arcsec(
                getattr(band_coords, lon),
                getattr(band_coords, lat),
                np.asarray(getattr(obs.coords, lon))[band_mask],
                np.asarray(getattr(obs.coords, lat))[band_mask],
            )
            assert err.max() < 1e-3, f"{band_name} {lon}/{lat}: max difference {err.max():.4f} arcsec"


def test_slice_does_not_keep_the_cached_center():
    """P5: a slice returned the cached centre of its parent (3 deg on the sky here)."""
    t = T0 + np.arange(0, 60, 1.0)
    offsets = np.radians(np.c_[np.linspace(-5, 5, 11), np.zeros(11)])

    coords = boresight(t, 100.0, 45.0).broadcast(offsets)
    coords.center("az/el")  # cache the centre of the parent
    for frame in ["az/el", "ra/dec"]:
        slice_center = coords[6:].center(frame)
        fresh_center = boresight(t, 100.0, 45.0).broadcast(offsets[6:]).center(frame)
        err = separation_arcsec(slice_center[0].rad, slice_center[1].rad, fresh_center[0].rad, fresh_center[1].rad)
        assert err < 0.1, f"{frame}: slice centre is {err:.1f} arcsec off"


def test_summary():
    """P13: Coordinates.summary used self.boresight without calling it, and raised AttributeError."""
    t = T0 + np.arange(0, 60, 1.0)
    summary = boresight(t, 100.0, 45.0).broadcast(grid_offsets(n=3)).summary

    assert list(summary.index) == ["az", "el", "ra", "dec"]
    assert summary.loc["el", "mean"].startswith("45.0")


@pytest.mark.parametrize("throw", [2.0, 10.0, 20.0])
def test_downsample_keeps_the_fit_reference(throw):
    """A downsampled subset fitted its transform around its own centre instead of the boresight (0.05-0.13")."""
    t = T0 + np.arange(0, 600, 0.5)
    az = 180.0 + throw * np.sin(2 * np.pi * t / 60.0)
    offsets = np.radians([[0.0, 0.0], [1.5, 1.5], [-1.5, 1.0]])
    sub = boresight(t, az, 50.0).broadcast(offsets)[1:]
    ds = sub.downsample(factor=2)
    n = min(ds.t.shape[-1], sub.t[::2].shape[-1])
    assert np.allclose(ds.t[:n], sub.t[::2][:n])

    err = separation_arcsec(ds.ra[:, :n], ds.dec[:, :n], sub.ra[:, ::2][:, :n], sub.dec[:, ::2][:, :n])
    assert err.max() < 1e-3, f"max difference {err.max():.4f} arcsec"
