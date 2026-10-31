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
    unjitted_phi_theta_to_xyz,
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

# spacing of the time nodes at which the frame transforms are computed exactly (the Earth turns 0.25 deg in 60 s)
_TRANSFORM_NODE_SPACING_SECONDS = 60.0


class Coordinates:
    """
    A class for managing coordinates, allowing us to access different coordinate frames at once.

    Azimuth and elevation are geometric (vacuum) positions: atmospheric refraction is not modelled, and the conversion
    to ra/dec (ICRS) and galactic coordinates matches astropy's AltAz frame with pressure=0. Real telescope pointing is
    usually given as apparent (refracted) positions, which differ by ~1' at an elevation of 30 degrees.

    The conversions to other frames are a 3x3 matrix fitted on fiducial points around the center of the coordinates
    (or around the boresight, for coordinates made by `broadcast` and their slices), computed with astropy every 60
    seconds of absolute time and interpolated in between. They agree with astropy to ~0.1" near that center while it
    stays within ~5 degrees over +-2 minutes. The error grows with the distance from the center (~0.3" at 10 degrees,
    ~1" at 20 degrees) and with the motion of the center: for back-and-forth azimuth scans at 30-70 degrees
    elevation it is up to ~0.5" for a +-10 degree throw, ~0.2-1.8" for +-20 degrees and ~2-12" for +-60 degrees
    (largest at low elevation), and time chunks of a +-20 degree scan differ by up to ~0.6".

    Offsets (see `offsets`, `broadcast`) are a zenithal equidistant projection with +y towards increasing latitude
    and +x towards decreasing longitude (x ~ -dphi * cos(theta)), so the az/el and ra/dec offset frames are mirror
    images of each other.
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

        # an optional reference track (t, phi, theta) in this frame, around which the frame transforms are fitted
        # (e.g. the boresight). Without one, the transforms are fitted around the center of these coordinates.
        self._fit_reference = None

    def initialize(self):
        """
        Do all the initial computations needed to convert to some other frame.
        """

        ref_time = ttime.monotonic()

        self.shaped_t = np.atleast_1d(self.t)
        fit_t, fit_phi, fit_theta = self._fit_reference or self._center_track(self._phi, self._theta)
        fit_center_phi_theta = np.c_[fit_phi, fit_theta]

        # (n_nodes) times at which the transform is computed with astropy: the points of an absolute time grid that
        # _interpolate_transform needs, so that the transform at a given time does not depend on the time span
        node_index = np.floor(self.shaped_t / _TRANSFORM_NODE_SPACING_SECONDS).astype(np.int64)
        self.fid_node_index = np.unique(np.unique(node_index)[:, None] + np.arange(-1, 3)[None])
        self.fid_t = self.fid_node_index * _TRANSFORM_NODE_SPACING_SECONDS

        sample_indices = interp1d(
            fit_t,
            np.arange(len(fit_t)),
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",
        )(self.fid_t).astype(int)

        # fiducial offsets around the center to train a transformation matrix: the center itself, and rings of
        # radius 1, 10 and 30 degrees. The fit is linear, so the fiducials must stay local: a frame change
        # (aberration, etc.) is not a rotation over the whole sphere.
        # shape: (n_fid, 2)
        psi = np.linspace(0, 2 * np.pi, 12 + 1)[:-1]
        fid_offsets = np.concatenate(
            [np.zeros((1, 2))] + [np.radians(r) * np.c_[np.cos(psi), np.sin(psi)] for r in [1, 10, 30]],
            axis=0,
        )

        # for sample_index =

        # the fit is done in float64 numpy, so that it is accurate and does not depend on the number of nodes
        fpt = unjitted_offsets_to_phi_theta(
            fid_offsets,
            np.float64(fit_center_phi_theta[:, 0][sample_indices])[..., None],
            np.float64(fit_center_phi_theta[:, 1][sample_indices])[..., None],
            xp=np,
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
        self.fid_points = {self.frame.name: unjitted_phi_theta_to_xyz(self.fid_phi, self.fid_theta, xp=np)}
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

        self.fid_points[frame.name] = unjitted_phi_theta_to_xyz(frame_fid_phi, frame_fid_theta, xp=np)

        # voodoo! (least squares, per node)
        self.transforms[frame.name] = np.linalg.inv(self.AT @ self.A) @ self.AT @ self.fid_points[frame.name]

        transform_stack = self._interpolate_transform(self.transforms[frame.name])

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

    def _interpolate_transform(self, node_transforms):
        """
        Interpolate the transforms computed at the nodes to the sample times, with a local cubic (four-point
        Lagrange) interpolation. The interpolated transform at some time only depends on the four nearest nodes.
        """
        u = np.atleast_1d(self.t) / _TRANSFORM_NODE_SPACING_SECONDS  # not shaped_t, which slices do not update
        k = np.floor(u).astype(np.int64)
        s = (u - k)[:, None, None]
        i = np.searchsorted(self.fid_node_index, k - 1)

        weights = [
            -s * (s - 1) * (s - 2) / 6,
            (s + 1) * (s - 1) * (s - 2) / 2,
            -(s + 1) * s * (s - 2) / 2,
            (s + 1) * s * (s - 1) / 6,
        ]

        transform_stack = sum(w * node_transforms[i + j] for j, w in enumerate(weights))

        return transform_stack if hasattr(self.t, "__len__") else transform_stack[0]

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

        # a subset fits its frame transforms around the center of the full coordinates, so that a subset (e.g. one
        # band) gets the same transformed coordinates as the full set
        if not self.initialized and self._fit_reference is None:
            clone._fit_reference = self._center_track(self._phi, self._theta)

        # cached metadata describe the parent, not the slice
        clone.centers = {}
        clone.__dict__.pop("summary", None)

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

        ds_t = np.arange(self.t.min(), self.t.max(), timestep)
        ds_phi = sp.interpolate.interp1d(self.t, self._phi, axis=-1, bounds_error=False, fill_value="extrapolate")(ds_t)

        ds_theta = sp.interpolate.interp1d(self.t, self._theta, axis=-1, bounds_error=False, fill_value="extrapolate")(ds_t)

        return Coordinates(
            t=ds_t,
            phi=ds_phi,
            theta=ds_theta,
            earth_location=self.earth_location,
            frame=self.frame.name,
            dtype=self.dtype,
        )._with_fit_reference(self._fit_reference or self._center_track(self._phi, self._theta))

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
        boresight = self.boresight()
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

    def _with_fit_reference(self, reference):
        """Fit the transforms around `reference` (e.g. that of the coordinates this was derived from)."""
        self._fit_reference = reference
        return self

    def _center_track(self, phi, theta):
        """
        The center of (phi, theta) at each time, as a reference track (t, phi, theta) for the frame transforms.
        """
        keep_dims = (-1,) if hasattr(self.t, "__len__") else ()
        return (np.atleast_1d(self.t), *get_center_phi_theta(phi, theta, keep_dims=keep_dims))

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
        coords = Coordinates(
            t=self.t,
            phi=phi_theta[..., 0],
            theta=phi_theta[..., 1],
            earth_location=self.earth_location,
            frame="az/el",
        )
        # fit the frame transforms around the boresight, so that they do not depend on which offsets are present
        coords._fit_reference = self._center_track(self.az, self.el)
        return coords

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
