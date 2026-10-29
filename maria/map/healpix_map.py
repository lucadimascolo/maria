import copy
import os

import dask.array as da
import dask.base
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ..coords import FRAMES, Coordinates, Frame
from ..units import Quantity
from .base import Map

here, this_filename = os.path.split(__file__)

# the frames that healpy can rotate between
_HEALPY_COORD = {"galactic": "G", "ra/dec": "C"}


class HEALPixMap(Map):
    """
    A spherical map. It has shape (stokes, nu, t, pix).
    """

    def __init__(
        self,
        data: float,
        weight: float = None,
        stokes: float = None,
        nu: float = None,
        t: float = None,
        z: float = None,
        v: float = None,
        pixels: float = None,
        beam: tuple[float, float, float] = 0.0,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        dtype: type = np.float32,
        enforce_valid_map_quantity: bool = True,
    ):
        if weight is not None:
            if weight.shape != data.shape:
                raise ValueError(f"'data' {data.shape} and 'weight' {weight.shape} should have the same shape.")
        else:
            weight = da.ones_like(data)

        if pixels is None:
            pixels = np.arange(data.shape[-1])

        self.pixels = pixels

        if not hp.pixelfunc.isnpixok(len(pixels)):
            raise ValueError(f"Invalid pixel count {len(pixels)} (must be of the form 12 * (2^k)^2).")

        self.nside = hp.pixelfunc.npix2nside(len(pixels))
        self.frame = Frame(frame)

        super().__init__(
            data=data,
            weight=weight,
            stokes=stokes,
            nu=nu,
            t=t,
            z=z,
            v=v,
            beam=beam,
            map_dims={"pixels": len(pixels)},
            units=units,
            frame=frame,
            degrees=degrees,
            dtype=dtype,
            enforce_valid_map_quantity=enforce_valid_map_quantity,
        )

        # if not hasattr(beam, "__len__"):
        #     beam = beam or self.resolution
        #     beam = (beam, beam, 0)

        # if len(beam) != 3:
        #     raise ValueError("'beam' must be either a number or a tuple of (major, minor, angle)")

        # self.beam = tuple(np.radians(beam) if degrees else beam)

    def _pointing_matrix_ingredients(self, coords: Coordinates, bilinear: bool = False, zero_outside: bool = False):
        # a HEALPix map covers the whole sphere, so no sample is outside it

        n_pixels = self.dims["pixels"]
        n_samples = coords.size

        # sample the map in its own frame
        phi = np.asarray(getattr(coords, self.frame.phi_name)).ravel()
        theta = np.asarray(getattr(coords, self.frame.theta_name)).ravel()

        # with the same (-1, *coords.shape) layout as a ProjectionMap
        if bilinear:
            # the four nearest pixels, with healpy's bilinear interpolation weights
            pixels, weights = hp.pixelfunc.get_interp_weights(self.nside, theta=np.pi / 2 - theta, phi=phi)
        else:
            pixels = hp.pixelfunc.ang2pix(self.nside, theta=np.pi / 2 - theta, phi=phi)[None]
            weights = np.ones(pixels.shape)

        pixels = pixels.reshape(-1, *coords.shape)
        weights = weights.reshape(-1, *coords.shape)
        samples = np.arange(n_samples).reshape(coords.shape) + np.zeros_like(pixels)

        # add time index
        if self.dims.get("t", 0) > 1:
            pixels += (np.digitize(coords._t, bins=self.t_bins) - 1) * self.dims["pixels"]
            n_pixels *= self.dims["t"]

        return samples, pixels, weights, n_pixels, n_samples

    @property
    def resolution(self):
        return hp.pixelfunc.nside2resol(self.nside)

    @property
    def pixel_area(self):
        return Quantity(hp.pixelfunc.nside2resol(self.nside) ** 2, "sr")

    @property
    def npix(self):
        return self.dims["pixels"]

    def package(self, compute: bool = False):
        package = copy.deepcopy(
            {
                "data": self.data,
                "frame": self.frame.name,
                "units": self.units,
                "beam": self.beam.deg,
                "degrees": True,
            }
        )

        if compute:
            package["data"] = np.asarray(package["data"])

        if self._weight is not None:
            package["weight"] = self._weight
            if compute:
                package["weight"] = np.asarray(package["weight"])

        for dim in self.dims:
            package[dim] = getattr(self, dim)

        return package

    # @property
    # def X(self):
    #     return np.meshgrid(self.x_side, self.y_side)[0]

    # @property
    # def Y(self):
    #     return np.meshgrid(self.x_side, self.y_side)[1]

    def smooth(self, sigma: float = None, fwhm: float = None, inplace: bool = False, degrees: bool = True):
        """
        Smooth the map with a circular Gaussian beam.

        The spherical harmonic transform of the map (about 1.5 times the size of a float32 map) is kept on the map,
        so that smoothing it again with another beam is faster.
        """
        return self._smooth(sigma=sigma, fwhm=fwhm, inplace=inplace, degrees=degrees)

    def _smooth(
        self,
        sigma: float = None,
        fwhm: float = None,
        inplace: bool = False,
        degrees: bool = True,
        pol: bool = False,
        frame: str = None,
    ):
        """
        smooth(), which can also smooth Q and U as a spin-2 field (pol=True, as healpy.smoothing(pol=True)) instead
        of as scalar maps, and rotate the map (and its weight) to another frame ("ra/dec" or "galactic"), as the
        simulation does. Q and U are then referred to the north of that frame, which needs pol=True.
        """
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))

        if not isinstance(sigma, Quantity):
            sigma = Quantity(sigma, "deg" if degrees else "rad")

        # stokes is always the first dimension when present
        stokes = list(self.stokes) if "stokes" in self.dims else []

        frame = Frame(frame).name if frame is not None else self.frame.name
        if frame != self.frame.name:
            if not {frame, self.frame.name} <= _HEALPY_COORD.keys():
                raise ValueError(f"Cannot rotate a HEALPix map from '{self.frame.name}' to '{frame}'.")
            if ({"Q", "U"} & set(stokes)) and not (pol and {"Q", "U"} <= set(stokes)):
                raise ValueError("Rotating Q and U to another frame needs both of them and pol=True.")

        maps = np.asarray(self.data).reshape(max(len(stokes), 1), -1, self.npix)

        groups = [[i] for i in range(len(maps))]
        if pol and {"Q", "U"} <= set(stokes):
            # Q and U are a spin-2 field, smoothed together with I (or with a zero I if the map has none)
            iqu = [stokes.index(s) if s in stokes else None for s in "IQU"]
            groups = [iqu] + [[i] for i in range(len(maps)) if i not in iqu]

        smoothed_maps = np.zeros_like(maps)
        for k in range(maps.shape[1]):
            for group in groups:
                smoothed = self._smooth_slice(maps, group, k, sigma.radians, frame)
                for i, smoothed_map in zip(group, smoothed):
                    if i is not None:
                        smoothed_maps[i, k] = smoothed_map

        data = smoothed_maps.reshape(self.data.shape)

        beam, beam_area_ratio = self._smoothed_beam(sigma)
        data = data * beam_area_ratio

        weight = self.weight
        if frame != self.frame.name:
            rotator = hp.Rotator(coord=[_HEALPY_COORD[self.frame.name], _HEALPY_COORD[frame]])
            weight = np.stack([rotator.rotate_map_pixel(w) for w in np.asarray(weight).reshape(-1, self.npix)])
            weight = weight.reshape(self.weight.shape).astype(self.weight.dtype)

        if inplace:
            self.data = da.asarray(data)
            self.weight = da.asarray(weight)
            self.beam = Quantity(beam, "deg")
            self.frame = Frame(frame)

        else:
            package = self.package()
            package.update({"data": data, "weight": weight, "beam": beam, "frame": frame})
            return type(self)(**package)

    def _data_key(self):
        """
        Identifies the content of the data: the token of a dask array is its name, which changes when it is edited in
        place, and that of a numpy array (e.g. data assigned to the map) hashes its values.
        """
        return dask.base.tokenize(self.data)

    def _smooth_slice(self, maps, indices, k, sigma, frame):
        """
        healpy.smoothing of the maps[indices, k] (a spin-2 field if there are three of them; with healpy's default
        lmax = 3 nside - 1 and iter = 3), rotated to the frame, with the (rotated) spherical harmonic transform cached
        on the map, so that smoothing with several beams (e.g. one per band) does it once. A Gaussian beam is
        isotropic, so smoothing and rotating commute.
        """
        if getattr(self, "_alm_cache_data", None) != self._data_key():
            self._alm_cache, self._alm_cache_data = {}, self._data_key()

        lmax = 3 * self.nside - 1
        pol = len(indices) == 3
        key = (tuple(indices), k, frame)
        if key not in self._alm_cache:
            m = np.stack([maps[i, k] if i is not None else np.zeros(self.npix, dtype=maps.dtype) for i in indices])
            m[np.isnan(m)] = hp.UNSEEN  # NaN pixels are masked, like healpy's UNSEEN ones (which it sets to zero)
            alm = hp.sphtfunc.map2alm(m if pol else m[0], lmax=lmax, iter=3, pol=pol)
            if frame != self.frame.name:
                alm = hp.Rotator(coord=[_HEALPY_COORD[self.frame.name], _HEALPY_COORD[frame]]).rotate_alm(alm, lmax=lmax)
            self._alm_cache[key] = alm
        alm = hp.sphtfunc.smoothalm(self._alm_cache[key], sigma=sigma, pol=pol, inplace=False)
        return np.atleast_2d(hp.sphtfunc.alm2map(alm, nside=self.nside, lmax=lmax, pol=pol))

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=float, data=self.data)

            if self._weight is not None:
                f.create_dataset("weight", dtype=float, data=self._weight)

            f.create_dataset("frame", data=self.frame.name)

            for field in ["stokes", "nu", "t", "units"]:
                f.create_dataset(field, data=getattr(self, field))

    def plot(self):
        for i in range(len(self.stokes)):
            m = self.data[i, 0, 0]
            min, max = da.percentile(m, q=[0.1, 99.9])

            hp.newvisufunc.projview(
                m.compute(),
                min=min,
                max=max,
                cmap="cmb",
                sub=(len(self.stokes), 1, 1 + i),
            )
            # hp.visufunc.gnomview(self.data[i,0,0], cmap="cmb")

        plt.tight_layout()

    def __repr__(self):
        return f"""{self.__class__.__name__}:
{self.__repr_base__()}
  pixels({self.npix}):
    nside: {self.nside}
    resolution: {Quantity(self.resolution, "rad")}
  frame: {self.frame.name}
  beam(maj, min, rot): {self.beam_repr()}
  memory: {Quantity(self.data.nbytes + self.weight.nbytes, "B")}"""
