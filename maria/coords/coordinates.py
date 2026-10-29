from __future__ import annotations

import functools
import logging
import os
import time as ttime
from copy import deepcopy

import arrow
import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.interpolate import interp1d

from ..io import DEFAULT_TIME_FORMAT, humanize_time, repr_lat_lon, repr_phi_theta
from ..units import Quantity
from ..utils import get_utc_day_hour, get_utc_year_day
from .frame import FRAMES, Frame
from .transforms import (
    get_center_phi_theta,
    offsets_to_phi_theta,
    phi_theta_to_offsets,
    phi_theta_to_xyz,
    unjitted_offsets_to_phi_theta,
    xyz_to_phi_theta,
)

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)
frames = pd.read_csv(f"{here}/frames.csv", index_col=0).T

DEFAULT_EARTH_LOCATION = EarthLocation.from_geodetic(
    0.0,
    90.0,
    height=0.0,
    ellipsoid=None,
)  # noqa
DEFAULT_TIMESTAMP = arrow.now().timestamp()


def _unwrap_longitude(phi):
    """Unwrap the longitude along the last axis, ignoring NaNs (a NaN does not propagate along the track).

    Returns the unwrapped longitude, a boolean array over the leading axes that says which tracks cross the wrap, and
    the maximum (finite) longitude of each track."""
    phi = np.asarray(phi)
    n = phi.shape[-1]
    finite = np.isfinite(phi)
    # compare each sample with the last finite sample before it
    last_finite = np.maximum.accumulate(np.where(finite, np.arange(n), 0), axis=-1)
    filled = np.take_along_axis(phi, last_finite, axis=-1)
    step = np.diff(filled, axis=-1)
    with np.errstate(invalid="ignore"):
        jump = np.where(np.abs(step) > np.pi, -2 * np.pi * np.round(step / (2 * np.pi)), 0.0)
    crosses = (jump != 0).any(axis=-1)
    if not crosses.any():
        return phi, crosses, np.zeros(crosses.shape)
    correction = np.concatenate([np.zeros((*phi.shape[:-1], 1)), np.cumsum(jump, axis=-1)], axis=-1)
    with np.errstate(invalid="ignore"):
        phi_max = np.max(np.where(finite, phi, -np.inf), axis=-1)
    return np.where(crosses[..., None], phi + correction, phi), crosses, phi_max


class Coordinates:
    """
    A class for managing coordinates, allowing us to access different coordinate frames at once.
    """

    def __init__(
        self,
        phi: float = 0.0,
        theta: float = 0.0,
        r: float = 0.0,
        t: float = DEFAULT_TIMESTAMP,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        earth_location: EarthLocation = DEFAULT_EARTH_LOCATION,
        frame: str = "az/el",
        dtype: type = np.float64,
    ):
        self.earth_location = earth_location
        self.frame = Frame(frame)
        self.dtype = dtype

        if isinstance(t, str):
            t = arrow.get(t).timestamp()

        # DO NOT BROADCAST TIME. IT STAYS ONE-DIMENSIONAL.
        for attr, value in zip(
            ["x", "y", "z", "r", "phi", "theta", "t"],
            np.broadcast_arrays(x, y, z, r, phi, theta, t),
        ):
            # if not isinstance(value, dask.array.Array):
            #     value = da.asarray(value)
            # setattr(self, f"_{attr}", da.asarray(value))
            setattr(self, f"_{attr}", value)

        setattr(self, self.frame.phi_name, self._phi)
        setattr(self, self.frame.theta_name, self._theta)

        if hasattr(t, "__len__"):
            for axis in range(len(self.t.shape) - 1):
                if (np.ptp(self.t, axis=axis) > 0).any():
                    raise ValueError("Only the last axis can vary in time.")

        self.initialized = False
        self.computed_frames = [self.frame.name]

        # dicts for caching metadata
        self.centers = {}

    def initialize(self):
        """
        Do all the initial computations needed to convert to some other frame.
        """

        ref_time = ttime.monotonic()

        self.shaped_t = np.atleast_1d(self.t)
        keep_dims = (-1,) if hasattr(self.t, "__len__") else ()
        t_ordered_center_phi_theta = np.c_[get_center_phi_theta(self._phi, self._theta, keep_dims=keep_dims)]

        # (nt) t samples on which to explicitly compute the transformation from astropy
        t_samples_min_res_seconds = 600
        t_samples_min = np.min(self.t) - 1e0
        t_samples_max = np.max(self.t) + 1e0
        n_t_samples = int(
            np.maximum(
                3,
                (t_samples_max - t_samples_min) / t_samples_min_res_seconds,
            ),
        )

        n_t_samples = np.minimum(n_t_samples, 10000)

        if 3 <= self.t.size < n_t_samples:
            self.fid_t = self.t
        else:
            self.fid_t = np.linspace(t_samples_min, t_samples_max, n_t_samples)

        sample_indices = interp1d(
            self.shaped_t,
            np.arange(len(self.shaped_t)),
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",
        )(self.fid_t).astype(int)

        # three fiducial offsets from the boresight to train a transformation matrix
        # shape: (n_fid, 2)
        fid_offsets = np.radians([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        psi = np.linspace(0, 2 * np.pi, 12 + 1)[:-1]
        fid_offsets = np.concatenate(
            [r * np.c_[np.cos(psi), np.sin(psi)] for r in [1, 10, 30]],
            axis=0,
        )

        # for sample_index =

        fpt = unjitted_offsets_to_phi_theta(
            fid_offsets[..., :2],
            t_ordered_center_phi_theta[:, 0][sample_indices][..., None],
            t_ordered_center_phi_theta[:, 1][sample_indices][..., None],
        )

        self.fid_phi, self.fid_theta = fpt[..., 0], fpt[..., 1]

        # self.fid_phi = np.zeros((n_t_samples, len(fid_offsets)))
        # self.fid_theta = np.zeros((n_t_samples, len(fid_offsets)))

        # for ii, it in enumerate(sample_indices):
        #     self.fid_phi[ii], self.fid_theta[ii] = offsets_to_phi_theta(
        #     fid_offsets[..., :1],
        #     *t_ordered_center_phi_theta[it],
        # )

        self.fid_skycoords = {
            self.frame.name: SkyCoord(
                self.fid_phi * u.rad,
                self.fid_theta * u.rad,
                obstime=Time(
                    self.fid_t[:, None],
                    format="unix",
                ),
                frame=frames[self.frame.name]["astropy_name"],
                location=self.earth_location,
            ),
        }

        self.transforms = {}
        self.fid_points = {self.frame.name: phi_theta_to_xyz(self.fid_phi, self.fid_theta)}
        self.A = self.fid_points[self.frame.name]
        self.AT = np.swapaxes(self.fid_points[self.frame.name], -2, -1)

        self.initialized = True

        duration_s = ttime.monotonic() - ref_time
        logger.debug(
            f"Initialized Coordinates with shape {self.shape} in {humanize_time(duration_s)}.",
        )  # noqa

    def compute_transform(self, frame):
        compute_transform_start_s = ttime.monotonic()

        frame = Frame(frame)

        if not self.initialized:
            self.initialize()

        self.fid_skycoords[frame.name] = getattr(
            self.fid_skycoords[self.frame.name],
            frame.astropy_name,
        )

        frame_fid_phi = getattr(
            self.fid_skycoords[frame.name],
            frame.astropy_phi,
        ).rad
        frame_fid_theta = getattr(
            self.fid_skycoords[frame.name],
            frame.astropy_theta,
        ).rad

        self.fid_points[frame.name] = phi_theta_to_xyz(frame_fid_phi, frame_fid_theta)

        # voodoo!
        self.transforms[frame.name] = (
            np.linalg.inv(self.AT @ self.A) @ self.AT @ phi_theta_to_xyz(frame_fid_phi, frame_fid_theta)
        )

        transform_stack = interp1d(
            self.fid_t,
            self.transforms[frame.name],
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
            axis=0,
        )(self.t)

        frame_phi, frame_theta = xyz_to_phi_theta(
            (np.expand_dims(self.compute_points(), -2) @ transform_stack).squeeze(),
        )

        setattr(self, frame.phi_name, frame_phi)
        setattr(self, frame.theta_name, frame_theta)

        self.computed_frames.append(frame.name)

        duration = ttime.monotonic() - compute_transform_start_s
        logger.debug(
            f"Computed transform to frame '{frame.name}' for Coordinates with shape {self.shape} "
            f"in {humanize_time(duration)}.",
        )  # noqa

    @property
    def shape(self):
        return self._phi.shape

    @property
    def size(self):
        return self._phi.size

    @property
    def ndim(self):
        return self._phi.ndim

    def __getattr__(self, attr):
        if attr == "t":
            return self._t[tuple(0 for _ in range(self.ndim - 1))]

        if attr in ["x", "y", "z", "r", "phi", "theta"]:
            return getattr(self, f"_{attr}")

        for frame_name, config in FRAMES.items():
            if attr in [config["phi"]["name"], config["theta"]["name"]]:
                self.compute_transform(frame=frame_name)
                return getattr(self, attr)

        raise AttributeError(f"'Coordinates' object has no attribute '{attr}'")

    def __getitem__(self, key):
        clone = deepcopy(self)
        attrs = ["_x", "_y", "_z", "_r", "_phi", "_theta", "_t"]
        attrs.extend([frames[frame][c] for frame in self.computed_frames for c in ["phi", "theta"]])
        for attr in attrs:
            setattr(clone, attr, getattr(clone, attr)[key])

        return clone

    @property
    def day_hour(self):
        return np.array(list(map(get_utc_day_hour, self.t)))

    @property
    def year_day(self):
        return np.array(list(map(get_utc_year_day, self.t)))

    @property
    def timestep(self):
        if len(self.t):
            return np.mean(np.gradient(self.t))
        return None

    def downsample(self, timestep: float = None, factor: int = None):
        if timestep is None and factor is None:
            raise ValueError("You must supply either 'timestep' or 'factor'.")

        timestep = timestep or factor * self.timestep

        # the downsampled times must cover the whole span, so that upsampling never extrapolates
        n = int(np.ceil((self.t.max() - self.t.min()) / timestep))
        ds_t = self.t.min() + timestep * np.arange(n + 1)
        ds_t[-1] = max(ds_t[-1], self.t.max())  # guard against rounding at unix-time magnitudes
        return self._interpolate(ds_t)

    def _interpolate(self, t):
        """The coordinates interpolated (and extrapolated) to the times t."""
        # interpolate the longitude across the 0/2π wrap: only the tracks that cross it are unwrapped (and wrapped
        # back into their own range afterwards), so the others are interpolated exactly as before. A track that crosses
        # the wrap is in [-π, π) if it never exceeds π, and in [0, 2π) otherwise: per-detector azimuths near north can
        # dip slightly below 0 while the boresight is in [0, 2π)
        phi, crosses, phi_max = _unwrap_longitude(self._phi)
        ds_phi = sp.interpolate.interp1d(self.t, phi, axis=-1, bounds_error=False, fill_value="extrapolate")(t)
        if crosses.any():
            lower = np.where(phi_max <= np.pi, -np.pi, 0.0)[crosses]
            ds_phi[crosses] = (ds_phi[crosses] - lower[..., None]) % (2 * np.pi) + lower[..., None]

        ds_theta = sp.interpolate.interp1d(self.t, self._theta, axis=-1, bounds_error=False, fill_value="extrapolate")(t)

        return Coordinates(
            t=t,
            phi=ds_phi,
            theta=ds_theta,
            earth_location=self.earth_location,
            frame=self.frame.name,
            dtype=self.dtype,
        )

    def boresight(self):
        cphi, ctheta = get_center_phi_theta(self._phi, self.theta, keep_dims=(-1,))

        return Coordinates(
            t=self.t,
            phi=cphi,
            theta=ctheta,
            earth_location=self.earth_location,
            frame=self.frame.name,
            dtype=self.dtype,
        )

    @functools.cached_property
    def summary(self):
        # compute summary for the string repr
        summary = pd.DataFrame(columns=["min", "mean", "max"])
        boresight = self.boresight
        for attr in ["az", "el", "ra", "dec"]:
            for stat in ["min", "mean", "max"]:
                summary.loc[attr, stat] = f"{float(np.degrees(getattr(getattr(boresight, attr), stat)())):.03f}°"

        return summary

    @property
    def xyz(self):
        return np.concatenate([self.x[..., None], self.y[..., None], self.z[..., None]], axis=-1)

    def project(self, z, frame="az/el"):
        frame = Frame(frame)

        phi = getattr(self, frame.phi_name)
        theta = getattr(self, frame.theta_name)

        tan_theta = np.tan(theta)[..., None]
        p = (z - self.z)[..., None] * np.concatenate(
            [
                np.cos(phi)[..., None] / tan_theta,
                np.sin(phi)[..., None] / tan_theta,
                np.ones((*phi.shape, 1)),
            ],
            axis=-1,
        )

        return p + self.xyz

    def compute_points(self):
        return phi_theta_to_xyz(self._phi, self._theta)

    def center(self, frame: str = None):
        frame = Frame(frame or self.frame)
        if frame.name not in self.centers:
            center_phi, center_theta = np.array(
                get_center_phi_theta(
                    getattr(self, frame.phi_name),
                    getattr(self, frame.theta_name),
                )
            )
            self.centers[frame.name] = (Quantity(center_phi, "rad"), Quantity(center_theta, "rad"))
        return self.centers[frame.name]

    def hull(self, frame: str, center=None, lazy: bool = None):
        center = center if center is not None else self.center(frame=frame)
        offsets = self.offsets(frame=frame, center=center).reshape(-1, 2)

        lazy = lazy if lazy is not None else len(offsets) > 20000
        if lazy:
            offsets = offsets[np.random.choice(a=len(offsets), size=20000)]

        hull = sp.spatial.ConvexHull(offsets)

        return offsets[hull.vertices]

    def broadcast(self, offsets, frame="az/el"):
        phi_theta = unjitted_offsets_to_phi_theta(offsets[..., None, :], self.az, self.el)
        return Coordinates(
            t=self.t,
            phi=phi_theta[..., 0],
            theta=phi_theta[..., 1],
            earth_location=self.earth_location,
            frame="az/el",
        )

    def offsets(self, frame, center=None, compute: bool = False):
        offsets_s = ttime.monotonic()

        frame = Frame(frame)

        if center is None:
            center = self.center(frame=frame)

        center = (Quantity(center[0], "rad"), Quantity(center[1], "rad"))

        if compute:
            return self.offsets(frame=frame, center=center, compute=False)

        if isinstance(center, str):
            if center == "auto":
                center = self.center(frame=frame)
        if frame.name == "az/el":
            X = phi_theta_to_offsets(np.stack([self.az, self.el], axis=-1), center[0].rad, center[1].rad)
        elif frame.name == "ra/dec":
            X = phi_theta_to_offsets(np.stack([self.ra, self.dec], axis=-1), center[0].rad, center[1].rad)
        elif frame.name == "galactic":
            X = phi_theta_to_offsets(np.stack([self.l, self.b], axis=-1), center[0].rad, center[1].rad)

        logger.debug(
            f"Computed offsets for Coordinates with shape {self.shape} in {humanize_time(ttime.monotonic() - offsets_s)}."
        )

        return X

    def spread(self, frame="ra/dec"):
        dX = self.offsets(frame=frame)
        return dX.std(axis=list(range(1, dX.ndim)))

    def __repr__(self):
        lon = self.earth_location.lon.deg
        lat = self.earth_location.lat.deg

        date_string = arrow.get(np.mean(self.t)).to("utc").format(DEFAULT_TIME_FORMAT)

        return f"Coordinates(shape={self.shape}, earth_location={repr_lat_lon(lat, lon)}, time='{date_string} UTC')"

    @property
    def longitude(self):
        return Quantity(self.earth_location.lon.deg, "deg")

    @property
    def latitude(self):
        return Quantity(self.earth_location.lat.deg, "deg")

    def __repr__(self):
        repr_lat, repr_lon = repr_lat_lon(self.latitude.degrees, self.longitude.degrees)
        center_repr = "center:"
        for frame in set([self.frame.name, *list(self.centers)]):
            for key, value in repr_phi_theta(*self.center(frame), frame=frame).items():
                center_repr += f"\n    {key}: {value}"

        min_time = arrow.get(self.t.min())
        max_time = arrow.get(self.t.max())

        return f"""Coordinates:
  shape: {self.shape}
  {center_repr}
  location:
    lat: {repr_lat}
    lon: {repr_lon}
  time:
    min: {min_time.timestamp():.03f} ({min_time.format(DEFAULT_TIME_FORMAT)})
    max: {max_time.timestamp():.03f} ({max_time.format(DEFAULT_TIME_FORMAT)})"""
