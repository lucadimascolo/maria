from __future__ import annotations

import numpy as np
import pytest
import scipy as sp
from maria.coords import Coordinates


@pytest.mark.parametrize("timestep", [0.1, 0.3, 0.5, 1.0])
def test_downsample_covers_the_span(timestep):
    """The atmosphere is simulated on a downsampled time grid and interpolated to the TOD, so the grid must cover the
    whole TOD, also when the span is a multiple of the timestep at unix-time magnitudes (rounding)."""
    t = 1768449600.0 + np.arange(600) / 10
    coords = Coordinates(t=t, phi=np.zeros(t.size), theta=np.full(t.size, 1.0), frame="az/el")
    ds_t = coords.downsample(timestep=timestep).t
    assert ds_t.min() <= t.min() and ds_t.max() >= t.max()


@pytest.mark.parametrize("factor", [1.5, 2])
def test_downsample_across_the_azimuth_wrap(factor):
    """An azimuth track that crosses 0 must be interpolated across the wrap, not through all the other azimuths."""
    t = 1768449600.0 + np.arange(600) / 10

    def track(t):  # 0.1 deg/s, crossing az = 0 at a different sample for each of 8 detectors
        return np.radians(0.1 * (t[None] - t[0] - 30) + 0.0125 * np.arange(8)[:, None])

    coords = Coordinates(t=t, phi=track(t) % (2 * np.pi), theta=np.full((8, t.size), 1.0), frame="az/el")
    ds = coords.downsample(factor=factor)
    error = (ds.az - track(ds.t) + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(error).max() < 1e-6


def test_downsample_rewraps_only_the_tracks_that_cross():
    """Only the detectors whose longitude crosses the wrap are unwrapped: the others keep master's interpolation and
    range (here [-π, π)), and the crossing ones are wrapped back into their own range."""
    t = 1768449600.0 + np.arange(600) / 10
    still = np.radians(np.full(t.size, -20.0))
    crossing = np.radians(175 + 0.3 * (t - t[0]))
    phi = np.stack([still, (crossing + np.pi) % (2 * np.pi) - np.pi])
    ds = Coordinates(t=t, phi=phi, theta=np.full((2, t.size), 1.0), frame="az/el").downsample(factor=7)
    master = sp.interpolate.interp1d(t, phi[0], bounds_error=False, fill_value="extrapolate")(ds.t)
    assert np.array_equal(ds._phi[0], master.astype(ds._phi.dtype))
    assert ds._phi[1].min() >= -np.pi and ds._phi[1].max() < np.pi
    error = (ds._phi[1] - np.radians(175 + 0.3 * (ds.t - t[0])) + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(error).max() < 1e-6


def test_downsample_wrap_with_a_nan():
    """A NaN in a track that crosses the wrap affects only the downsampled samples next to it."""
    t = 1768449600.0 + np.arange(600) / 10
    track = np.radians(355 + 0.3 * (t - t[0]))
    phi = np.where(np.arange(t.size) == 5, np.nan, track % (2 * np.pi))
    ds = Coordinates(t=t, phi=phi, theta=np.full(t.size, 1.0), frame="az/el").downsample(factor=7)
    bad = ~np.isfinite(ds._phi)
    assert bad.sum() <= 2 and (np.abs(ds.t[bad] - t[5]) < 0.7).all()
    error = (ds._phi[~bad] - np.radians(355 + 0.3 * (ds.t[~bad] - t[0])) + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(error).max() < 1e-6


def test_downsample_wrap_keeps_the_azimuth_range_near_north():
    """Per-detector azimuths around a boresight that crosses north in [0, 2π) can dip slightly below 0. The crossing
    tracks must come back in [0, 2π) like the boresight, not in [-π, π); tracks that do not cross are bit-identical."""
    t = 1768449600.0 + np.arange(600) / 10
    boresight = np.radians(359 + 0.05 * (t - t[0])) % (2 * np.pi)
    offsets = np.radians([-0.2, -0.1, 0.0, 0.1, 0.2])
    crossing = boresight + offsets[:, None]  # no wrapping of the offsets: values in [-0.2°, 360.2°)
    away = np.radians(np.full((2, t.size), [[90.0], [270.0]]))
    phi = np.concatenate([crossing, away])
    ds = Coordinates(t=t, phi=phi, theta=np.full(phi.shape, 1.0), frame="az/el").downsample(factor=4)
    assert (crossing.min(axis=1)[:2] < 0).all()
    assert (ds._phi[:5] >= 0).all() and (ds._phi[:5] < 2 * np.pi).all()
    error = (ds._phi[:5] - np.radians(359 + 0.05 * (ds.t - t[0])) - offsets[:, None] + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(error).max() < 1e-6
    master = sp.interpolate.interp1d(t, phi[5:], axis=-1, bounds_error=False, fill_value="extrapolate")(ds.t)
    assert np.array_equal(ds._phi[5:], master.astype(ds._phi.dtype))
