import copy
import logging
import os
from typing import Callable, Iterable

import astropy as ap
import dask.array as da
import h5py
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.wcs import WCS

from ..array import Array
from ..coords import Coordinates, Frame, get_center_phi_theta, offsets_to_phi_theta, phi_theta_to_offsets
from ..io import FITS_DEFAULT_UNITS, FITS_TYPE_ALIASES, repr_phi_theta
from ..units import Quantity, parse_units
from ..utils import compute_pointing_matrix_ingredients
from .base import Map

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class ProjectionMap(Map):
    """
    A rectangular map projected on the sphere. It has shape (stokes, nu, t, y, x).

    Q and U follow the COSMO (healpy) convention; FITS files with POLCCONV = 'IAU' have their U flipped when loaded.
    """

    def __init__(
        self,
        data: float,
        units: str,
        weight: float = None,
        stokes: float = None,
        nu: Iterable[float] = None,
        t: Iterable[float] = None,
        v: Iterable[float] = None,
        z: Iterable[float] = None,
        eta: Iterable[float] = None,
        xi: Iterable[float] = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        x_res: float = None,
        y_res: float = None,
        xi_res: float = None,
        eta_res: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        beam: tuple[float, float, float] = 0.0,
        frame: str = "ra/dec",
        degrees: bool = True,
        dtype: type = np.float32,
        enforce_valid_map_quantity: bool = True,
    ):
        # give it five dimensions

        if x_res is not None:
            xi_res = x_res
            logger.warning("'x_res' is deprecated and will be removed in a future version (use 'xi_res' instead)")

        if y_res is not None:
            eta_res = y_res
            logger.warning("'y_res' is deprecated and will be removed in a future version (use 'eta_res' instead)")

        if data.ndim < 2:
            raise ValueError("'data' must be at least two-dimensional")

        if weight is not None:
            if weight.shape != data.shape:
                raise ValueError(f"'data' {data.shape} and 'weight' {weight.shape} should have the same shape.")

        map_dims = {"eta": data.shape[-2], "xi": data.shape[-1]}

        try:
            assert len(center) == 2
            self.center = tuple([*Quantity(center, ("deg" if degrees else "rad")).pin("deg")])
        except Exception:
            raise ValueError("'center' must be a two-tuple of numbers")

        super().__init__(
            data=data,
            weight=weight,
            stokes=stokes,
            nu=nu,
            t=t,
            z=z,
            v=v,
            beam=beam,
            map_dims=map_dims,
            units=units,
            frame=frame,
            degrees=degrees,
            dtype=dtype,
            enforce_valid_map_quantity=enforce_valid_map_quantity,
        )

        # from the inputs, construct xi and eta

        if all(x is None for x in [width, height, resolution, xi_res, eta_res]):
            if (xi is None) or (eta is None):
                raise ValueError("You must pass at least one of 'width', 'height', 'resolution', 'xi_res', or 'eta_res'.")

        else:
            xi_sign = np.sign(xi_res or 1)
            eta_sign = np.sign(eta_res or 1)

            if (width is not None) or (height is not None):
                if width is not None:
                    xi_res = xi_sign * width / (data.shape[-1] - 1)
                    if height is None:
                        eta_res = eta_sign * abs(xi_res)

                if height is not None:
                    eta_res = eta_sign * height / (data.shape[-2] - 1)
                    if width is None:
                        xi_res = xi_sign * abs(eta_res)

            if resolution is not None:
                xi_res = xi_sign * resolution
                eta_res = eta_sign * resolution

            xi = xi_res * (map_dims["xi"] - 1) * np.linspace(-0.5, 0.5, self.dims["xi"])
            eta = eta_res * (map_dims["eta"] - 1) * np.linspace(-0.5, 0.5, self.dims["eta"])

        self.xi = Quantity(xi, "deg" if degrees else "rad")
        self.eta = Quantity(eta, "deg" if degrees else "rad")

        # apply parity convention
        # parity_signature = {dim: (-1 if dim in ["v", "eta"] else 1) for dim in self.dims}
        # self.apply_parity(**parity_signature)

    def _pointing_matrix_ingredients(self, coords: Coordinates, bilinear: bool = False, zero_outside: bool = False):
        """
        zero_outside: give samples beyond the outer pixel edges (half a pixel past the outer pixel centres) of the map
        zero weight, i.e. no signal, instead of the value of the nearest edge pixel. Samples before or after the time
        range of the map always take its first or last epoch.
        """
        offsets = coords.offsets(center=(self.center[0].rad, self.center[1].rad), frame=self.frame.name)

        samples, pixels, weights, n_pixels, n_samples = compute_pointing_matrix_ingredients(
            x_list=(
                coords._t,
                offsets[..., 1],
                offsets[..., 0],
            ),
            side_list=(self.t.seconds, self.eta.radians, self.xi.radians),
            bilinear=bilinear,
        )

        if zero_outside:
            # (the offsets can be a float32 jax array: the mask is made in numpy, so that the weights keep their dtype)
            inside = np.ones(offsets.shape[:-1], dtype=bool)
            for x, side in [(np.asarray(offsets[..., 1]), self.eta.radians), (np.asarray(offsets[..., 0]), self.xi.radians)]:
                if np.size(side) > 1:
                    # the outer pixel edges, which also works for a descending side
                    first_edge = side[0] - 0.5 * (side[1] - side[0])
                    last_edge = side[-1] + 0.5 * (side[-1] - side[-2])
                    inside &= (x - first_edge) * (x - last_edge) <= 0
            weights = weights * inside.reshape(weights.shape[1:])

        return samples, pixels, weights, n_pixels, n_samples

    def header(self):

        averaged_beam = self.beam.mean(axis=tuple(range(len(self.dims) - 2)))

        header = ap.io.fits.header.Header()
        header["SIMPLE"] = "T / conforms to FITS standard"
        header["BITPIX"] = "-32 / array data type"
        header["NAXIS"] = len(self.dims)
        for dim_index, n in enumerate(list(self.dims.values())[::-1]):
            header[f"NAXIS{dim_index + 1}"] = n

        header["BUNIT"] = self.units
        header["BMAJ"] = averaged_beam[0].degrees.item()
        header["BMIN"] = averaged_beam[1].degrees.item()
        header["BPA"] = averaged_beam[2].degrees.item()

        CTYPE1 = self.frame.fits_phi
        header["CTYPE1"] = f"{CTYPE1}{(5 - len(CTYPE1)) * '-'}ARC"  # maria maps are zenithal equidistant
        header["CRVAL1"] = self.center[0].deg
        # the 1-based FITS pixel of the centre (zero offset), which a sliced map need not have in its middle
        header["CRPIX1"] = (self.xi.size + 1) / 2 - (self.xi[0].rad + self.xi[-1].rad) / (2 * self.xi_res.rad)
        header["CDELT1"] = -self.xi_res.deg  # longitude goes the other way
        header["CUNIT1"] = "deg"

        CTYPE2 = self.frame.fits_theta
        header["CTYPE2"] = f"{CTYPE2}{(5 - len(CTYPE2)) * '-'}ARC"
        header["CRVAL2"] = self.center[1].deg
        header["CRPIX2"] = (self.eta.size + 1) / 2 - (self.eta[0].rad + self.eta[-1].rad) / (2 * self.eta_res.rad)
        header["CDELT2"] = self.eta_res.deg
        header["CUNIT2"] = "deg"

        for dim_index, dim in enumerate(list(self.dims)[:-2][::-1]):
            AXIS = dim_index + 3

            dim_values = getattr(self, dim)
            units = FITS_DEFAULT_UNITS[dim]

            if dim == "stokes":
                # the FITS Stokes codes are I = 1, Q = 2, U = 3, V = 4
                dim_values = np.array(["IQUV".index(s) + 1 for s in dim_values])

            if units:
                dim_values = dim_values.to(units)

            if dim_values.size > 1:
                grad = np.gradient(dim_values)
                delt = np.median(grad)
                if not grad.std() / np.abs(delt) < 1e-6:
                    raise RuntimeError("Cannot write irregular maps to FITS")
            else:
                delt = 1.0  # any non-zero value, so that the WCS matrix is not singular

            # CRVAL is the value at the first pixel, which is pixel 1 in FITS
            header[f"CTYPE{AXIS}"] = FITS_TYPE_ALIASES[dim][0]
            header[f"CRVAL{AXIS}"] = dim_values[0]
            header[f"CRPIX{AXIS}"] = 1
            header[f"CDELT{AXIS}"] = delt
            header[f"CUNIT{AXIS}"] = units

        return header

    # @property
    # def points(self):
    #     return np.stack(np.meshgrid(self.y_side, self.x_side, indexing="ij"), axis=-1)

    def __repr__(self):
        # beam_repr = self.beam
        center_repr = "center:"
        for key, value in repr_phi_theta(self.center[0].rad, self.center[1].rad, frame=self.frame.name).items():
            center_repr += f"\n    {key}: {value}"
        return f"""{self.__class__.__name__}:
{self.__repr_base__()}
  eta({self.dims["eta"]}):
    height: {self.height}
    res: {self.eta_res}
  xi({self.dims["xi"]}): 
    width: {self.width}
    res: {self.xi_res}
  frame: {self.frame.name}
  {center_repr}
  beam(maj, min, psi): {self.beam_repr()}
  memory: {Quantity(self.data.nbytes + (self._weight.nbytes if self._weight is not None else 0), "B")}"""

    def package(self, compute: bool = False):
        package = copy.deepcopy(
            {
                "data": self.data,
                "center": (self.center[0].deg, self.center[1].deg),
                "frame": self.frame.name,
                "units": self.units,
                "beam": self.beam.deg,
                "degrees": True,
            }
        )

        if compute:
            package["data"] = package["data"].compute()

        if self._weight is not None:
            package["weight"] = self._weight
            if compute:
                package["weight"] = package["weight"].compute()

        for dim in self.dims:
            package[dim] = getattr(self, dim)

        return package

    @property
    def pixel_area(self):
        return Quantity(abs(self.xi_res.rad * self.eta_res.rad), "sr")

    @property
    def resolution(self):
        if not np.isclose(self.xi_res.rad, self.eta_res.rad, rtol=1e-3):
            RuntimeError(
                "Cannot return attribute 'resolution'; ProjectionMap has x-resolution"
                f" {self.xi_res}° and y-resolution {self.eta_res}°."
            )
        return self.xi_res

    @property
    def width(self):
        return self.xi.max() - self.xi.min()

    @property
    def height(self):
        return self.eta.max() - self.eta.min()

    @property
    def n_xi(self):
        return len(self.xi)

    @property
    def xi_res(self):
        if not hasattr(self, "_xi_res"):
            xi_grad = np.gradient(self.xi.rad)
            med_xi_res = np.median(xi_grad)
            if np.std(xi_grad) / med_xi_res > 1e-4:
                self._xi_res = "irregular"
            else:
                self._xi_res = Quantity(med_xi_res, "rad")
        return self._xi_res

    @property
    def n_eta(self):
        return len(self.eta)

    @property
    def eta_res(self):
        if not hasattr(self, "_eta_res"):
            eta_grad = np.gradient(self.eta.rad)
            med_eta_res = np.median(eta_grad)
            if np.std(eta_grad) / med_eta_res > 1e-4:
                self._eta_res = "irregular"
            else:
                self._eta_res = Quantity(med_eta_res, "rad")
        return self._eta_res

    # @property
    # def xi_bins(self):
    #     """
    #     Following array indexing conventions,
    #     """
    #     self.xi_res.rad
    #     return self.width.rad * np.linspace(-0.5, 0.5, self.n_x + 1)

    # @property
    # def y_bins(self):
    #     """
    #     The negative is so that follows indexing rules
    #     """
    #     return -self.height.rad * np.linspace(-0.5, 0.5, self.n_y + 1)

    @property
    def x_side(self):
        logger.warning("Attribute 'x_side' is deprecated, use 'xi' instead")
        return self.xi.rad

    @property
    def y_side(self):
        logger.warning("Attribute 'y_side' is deprecated, use 'eta' instead")
        return self.eta.rad

    def resample(self, other_map):

        other_phi_theta = offsets_to_phi_theta(
            np.stack(np.meshgrid(other_map.xi.rad, other_map.eta.rad), axis=-1),
            other_map.center[0].rad,
            other_map.center[1].rad,
        )

        c = Coordinates(other_phi_theta[..., 0], other_phi_theta[..., 1], frame=other_map.frame)

        offsets = c.offsets(center=self.center, frame=self.frame)

        def interpolate(values):
            interpolator = sp.interpolate.RegularGridInterpolator(
                (self.eta.rad, self.xi.rad), np.moveaxis(values, (-2, -1), (0, 1)).compute(), bounds_error=False
            )
            return np.moveaxis(interpolator((offsets[..., 1], offsets[..., 0])), (0, 1), (-2, -1))

        # maps in units per pixel (like Jy/pixel) scale with the pixel area, as in reduce()
        pixel_area_ratio = float(other_map.pixel_area.sr / self.pixel_area.sr)
        pixel_area_factor = pixel_area_ratio ** -parse_units(self.units)["dimension_vector"]["pixel"]

        # the new map is on the other map's grid, so it is in the other map's frame
        return type(self)(
            interpolate(self.data) * pixel_area_factor,
            weight=interpolate(self._weight) * pixel_area_factor**-2 if self._weight is not None else None,
            units=self.units,
            center=other_map.center,
            frame=other_map.frame.name,
            **{dim: (getattr(other_map, dim) if dim in ["xi", "eta"] else getattr(self, dim)) for dim in self.dims},
            beam=self.beam,
        )

    def recenter(self):

        package = self.package()
        offsets = np.stack(np.meshgrid(self.xi.rad, self.eta.rad), axis=-1)
        phi_theta = offsets_to_phi_theta(offsets, self.center[0].rad, self.center[1].rad)
        new_center = get_center_phi_theta(phi_theta[..., 0], phi_theta[..., 1])
        new_offsets = phi_theta_to_offsets(phi_theta, float(new_center[0]), float(new_center[1]))
        package["xi_res"] = np.degrees(np.gradient(new_offsets[..., 0].mean(axis=0)).mean())
        package["eta_res"] = np.degrees(np.gradient(new_offsets[..., 1].mean(axis=1)).mean())
        package["center"] = (Quantity(new_center[0], "rad"), Quantity(new_center[1], "rad"))

        return type(self)(**package)

    def trim(self):
        trim_mask = (self.weight.compute() > 0) & np.isfinite(self.data.compute())
        return self[tuple([slice(indices.min(), indices.max() + 1, 1) for indices in np.where(trim_mask)])].recenter()

    def reduce(self, reduction: Iterable[int]):

        explicit_reduction = {
            dim: red for dim, red in zip(self.dims, [*(len(self.dims) - len(reduction)) * [1], *reduction])
        }
        reduction_shape = []
        dims_to_average = []
        trimming_slices = []

        new_dims = {}
        for dim_index, (dim, dim_len) in enumerate(self.dims.items()):
            red = explicit_reduction[dim]

            dim_trim_slice = slice(0, dim_len - dim_len % red, 1)
            dim_reduction_shape = (dim_len // red, red)

            if red > 1:
                if dim == "stokes":
                    raise ValueError("Cannot reduce stokes map dimension")
                new_dims[dim] = getattr(self, dim)[dim_trim_slice].reshape(dim_reduction_shape).mean(axis=-1)
            else:
                new_dims[dim] = getattr(self, dim)

            reduction_shape.extend(dim_reduction_shape)
            dims_to_average.append(2 * dim_index + 1)
            trimming_slices.append(dim_trim_slice)

        data = self.data[tuple(trimming_slices)].reshape(reduction_shape)
        weight = self.weight[tuple(trimming_slices)].reshape(reduction_shape)
        # pixels without data do not count (also in an unweighted map, whose weights are all one)
        weight = np.where(np.isfinite(data), weight, 0)

        # the weighted mean (the plain mean for an unweighted map, whose weights are all one)
        reduced_weight = np.nansum(weight, dims_to_average)
        reduced_data = np.nansum(data * weight, dims_to_average) / reduced_weight

        pixel_area_reduction = np.prod([explicit_reduction[dim] for dim in ["xi", "eta"]])
        pixel_area_factor = pixel_area_reduction ** -parse_units(self.units)["dimension_vector"]["pixel"]
        reduced_data *= pixel_area_factor

        return type(self)(
            data=reduced_data,
            weight=reduced_weight * pixel_area_factor**-2 if self._weight is not None else None,
            center=self.center,
            frame=self.frame.name,
            units=self.units,
            **new_dims,
            beam=self.beam,
        )

    def _pad_with_zero_sky(self, radius: float, offsets=None):
        """
        The map with zero sky (and, for a weighted map, the weight of the nearest edge pixel, as _smooth gives it with
        zero_outside) beyond its edges, out to at least `radius` (in radians) past the outer pixel centres. Given the
        (xi, eta) offsets (in radians) of the samples, it is padded only as far as they reach beyond the outer pixel
        centres (and at most `radius`): _smooth with zero_outside gives the pixels the same values either way.
        """
        package = self.package()
        pad_width = [(0, 0)] * (self.ndim - 2)
        for dim, offset_index in [("eta", 1), ("xi", 0)]:
            side = getattr(self, dim).deg
            n_pad = [0, 0]
            if side.size > 1:
                n_max = int(np.ceil(radius / abs(getattr(self, f"{dim}_res").rad)))
                n_pad = [n_max, n_max]
                if offsets is not None:
                    # the fractional pixel index of the samples, and the pixels (one more for bilinear sampling) that
                    # those beyond the outer pixel centres need
                    index = (np.degrees(np.asarray(offsets[..., offset_index])) - side[0]) / (side[1] - side[0])
                    reach = [-np.nanmin(index), np.nanmax(index) - (side.size - 1)]
                    n_pad = [int(np.clip(np.ceil(r) + 1, 0, n_max)) for r in reach]
            step = side[1] - side[0] if side.size > 1 else 0
            package[dim] = np.concatenate(
                [side[0] + step * np.arange(-n_pad[0], 0), side, side[-1] + step * np.arange(1, n_pad[1] + 1)]
            )
            pad_width.append(tuple(n_pad))

        package["data"] = da.pad(da.asarray(self.data), pad_width=pad_width, mode="constant", constant_values=0)
        if "weight" in package:
            package["weight"] = da.pad(da.asarray(package["weight"]), pad_width=pad_width, mode="edge")

        return type(self)(**package)

    def zero_pad(self):

        package = self.package()

        pad_width = [*len(self.slice_dims) * [(0, 0)], (1, 1), (1, 1)]
        package["data"] = np.pad(package["data"], pad_width=pad_width, constant_values=0).rechunk()

        if "weight" in package:
            package["weight"] = np.pad(package["weight"], pad_width=pad_width, constant_values=1).rechunk()

        # easier than resampling the dims
        for dim in ["xi", "eta"]:
            package.pop(dim)
            package[f"{dim}_res"] = getattr(self, f"{dim}_res")

        return type(self)(**package)

    def smooth(self, sigma: float = None, fwhm: float = None, degrees: bool = True):
        """
        Smooth the map with a circular Gaussian beam. The map is reflected about its edges.
        """
        return self._smooth(sigma=sigma, fwhm=fwhm, degrees=degrees)

    def _smooth(self, sigma: float = None, fwhm: float = None, degrees: bool = True, zero_outside: bool = False):
        """
        smooth(), where with zero_outside the sky beyond the edges of the map is zero, as the simulation samples it:
        the data are smoothed with zeros beyond the edges, and the weight with the nearest edge pixel (so the zero sky
        counts as much as the edge of the map).
        """
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))

        if not isinstance(sigma, Quantity):
            sigma = Quantity(sigma, "deg" if degrees else "rad")

        package = self.package()

        x_sigma_pixels = abs(sigma / self.xi_res)
        y_sigma_pixels = abs(sigma / self.eta_res)

        data_mode, weight_mode = ("constant", "nearest") if zero_outside else ("reflect", "reflect")
        sigma_pixels = (y_sigma_pixels, x_sigma_pixels)
        numer = sp.ndimage.gaussian_filter(self.data * self.weight, sigma=sigma_pixels, axes=(-2, -1), mode=data_mode)
        denom = sp.ndimage.gaussian_filter(self.weight, sigma=sigma_pixels, axes=(-2, -1), mode=weight_mode)

        with np.errstate(divide="ignore", invalid="ignore"):
            package["data"] = np.where(denom > 0, numer / denom, 0)

        if self._weight is not None:
            package["weight"] = denom

        package["beam"], beam_area_ratio = self._smoothed_beam(sigma)
        package["data"] = package["data"] * beam_area_ratio

        return type(self)(**package)

    def transfer_function(
        self,
        input_map=None,
        n_bins: int = 20,
        stokes: str = "I",
        slices=None,
        t_index: int = 0,
        window="hann",
        taper: float = 0.1,
        pad_factor: int = 1,
    ):
        """Compute the spatial transfer function relative to an input map.

        When the map has multiple frequency channels the transfer function is
        computed independently for each selected channel; ``result.T`` has shape
        ``(n_nu, n_bins)``.

        When the map was produced by a mapper whose TODs came from a
        :class:`~maria.Simulation` with ``map=...``, the input map is
        propagated automatically and this argument can be omitted.

        Parameters
        ----------
        input_map : ProjectionMap, optional
            The input sky map injected into the simulation.  Falls back to the
            map stored on this object (``_input_map``) when not given.
        n_bins : int
            Number of logarithmically-spaced spatial frequency bins.
        stokes : str
            Stokes parameter to use ("I", "Q", "U", or "V").
        slices : dict, optional
            Channel selection, e.g. ``dict(nu=[0, 2])``.  ``None`` includes all
            channels, consistent with how ``slices`` is used in ``.plot()``.
        t_index : int
            Time index for time-varying maps.
        window : str, bool, or np.ndarray
            Apodization window. ``"tukey"`` tapers only the edges;
            ``"hann"`` (default) applies a full Hann window; a 2D array of shape
            ``(ny, nx)`` is used directly; ``False`` disables windowing.
        taper : float
            Cosine-taper fraction for the Tukey window. Default is 0.1.
        pad_factor : int
            Zero-pad each axis to ``pad_factor`` times its length before the FFT,
            improving k-space sampling at large angular scales. Default is 1 (no padding).

        Returns
        -------
        TransferFunction
        """
        from .transfer import TransferFunction, compute_transfer_function

        if input_map is None:
            input_map = getattr(self, "_input_map", None)
        if input_map is None:
            raise ValueError(
                "No input map available. Pass input_map explicitly or run the simulation with map=<ProjectionMap>."
            )

        n_nu = self.dims["nu"] if "nu" in self.dims else 1
        nu_indices = list(np.atleast_1d(slices["nu"])) if (slices and "nu" in slices) else list(range(n_nu))

        rows = [
            compute_transfer_function(
                input_map,
                self,
                n_bins=n_bins,
                stokes=stokes,
                nu_index=i,
                t_index=t_index,
                window=window,
                taper=taper,
                pad_factor=pad_factor,
            )
            for i in nu_indices
        ]
        u = rows[0][0]
        T = np.stack([r[1] for r in rows])
        nu_qty = self.nu[nu_indices] if "nu" in self.dims else None

        # Per-channel beam FWHM: average over all slice axes except nu
        beam_fwhm = None
        if hasattr(self, "beam"):
            b_major = self.beam.rad[..., 0]  # (*slice_dims)
            if "nu" in self.slice_dims:
                nu_ax = list(self.slice_dims).index("nu")
                other = tuple(i for i in range(b_major.ndim) if i != nu_ax)
                all_fwhm = np.nanmean(b_major, axis=other) if other else b_major
            else:
                all_fwhm = np.full(n_nu, float(np.nanmean(b_major)))
            beam_fwhm = np.asarray(all_fwhm)[nu_indices]

        return TransferFunction(u=u, T=T, nu=nu_qty, beam_fwhm=beam_fwhm, input_map=input_map, output_map=self)

    def plot(
        self,
        slices: dict = {},
        attr: str = "data",
        cmap: str = "cmb",
        units: str = None,
        filename: str = None,
        contrast: str = 1e-3,
        center_zero: bool = False,
        vmin: float = None,
        vmax: float = None,
        rel_vmin: float = None,
        rel_vmax: float = None,
        ax_size: float = 5.0,
    ):

        if slices == "all":
            thick_slice_dims = {dim: n for dim, n in self.slice_dims.items() if n > 1}
            if len(thick_slice_dims) > 2:
                raise ValueError(
                    "Cannot plot all slices (map has more than two slice dimensions with size greater than one)"
                )

            slices = {k: np.expand_dims(np.arange(n), i) for i, (k, n) in enumerate(thick_slice_dims.items()) if n > 1}

        for dim in slices:
            if dim not in self.dims:
                raise ValueError(f"Map has no dimension '{dim}'")

        rel_vmin = rel_vmin or contrast
        rel_vmax = rel_vmax or 1.0 - contrast

        dim_slices = {dim: slices.get(dim, [0]) for dim in self.dims}
        dim_slices["eta"] = slice(None, None, None)
        dim_slices["xi"] = slice(None, None, None)

        SLICES = {dim: np.atleast_2d(x) for dim, x in zip(self.dims, np.broadcast_arrays(*dim_slices.values()))}

        if SLICES["xi"].ndim > 2:
            raise ValueError(f"Broadcasted slices have more than two dimensions")

        nrows = min(SLICES["xi"].shape)
        ncols = max(SLICES["xi"].shape)

        if units is None:
            units = Quantity(self.data, self.units).human_units
            logger.debug(f"Plotting with units '{units}'")

        plot_map = self.to(units)

        if attr == "data":
            map_data = plot_map.data
            map_weight = plot_map.weight
            map_units_repr = plot_map.u["math_name"]

        elif attr == "weight":
            map_data = plot_map.weight
            map_weight = da.ones_like(map_data)
            map_units_repr = f"({plot_map.u['math_name']})^{{-2}}"

        else:
            raise ValueError("'attr' must be either 'data' or 'weight'")

        # u = parse_units(units)

        grid_hu = Quantity(np.r_[self.xi.rad, self.eta.rad], "rad").hu

        lon_frame = f"{Frame(self.frame).fits['phi']}"
        lat_frame = f"{Frame(self.frame).fits['theta']}"

        header = fits.header.Header()
        header["CDELT1"] = -np.degrees(grid_hu["base_units_factor"])
        header["CDELT2"] = np.degrees(grid_hu["base_units_factor"])
        header["CRPIX1"] = 1
        header["CRPIX2"] = 1
        header["CTYPE1"] = f"{lon_frame}{(5 - len(lon_frame)) * '-'}SIN"
        header["CUNIT1"] = "deg     "
        header["CTYPE2"] = f"{lat_frame}{(5 - len(lat_frame)) * '-'}SIN"
        header["CUNIT2"] = "deg     "
        header["CRVAL1"] = self.center[0].deg
        header["CRVAL2"] = self.center[1].deg

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ax_size * ncols * 1.2, ax_size * nrows),
            subplot_kw={"projection": WCS(header)},
            constrained_layout=True,
            sharex=False,
            sharey=False,
            # ppi=256,
        )
        axes = np.atleast_2d(axes).reshape(SLICES["xi"].shape)

        for i in range(SLICES["xi"].shape[-2]):
            for j in range(SLICES["xi"].shape[-1]):
                ax = axes[i, j]
                ax_slices = {dim: SLICES[dim][i, j] for dim in self.dims}

                if "stokes" in ax_slices:
                    stokes = ax_slices["stokes"]
                    if isinstance(stokes, str):
                        if stokes not in self.stokes:
                            raise ValueError(f"Map does not have stokes parameter '{stokes}'")
                        ax_slices["stokes"] = list(self.stokes).index(stokes)

                map_slice_data = map_data[tuple(ax_slices.values())].compute()
                map_slice_weight = map_weight[tuple(ax_slices.values())].compute()

                if vmin is None or vmax is None:
                    if attr == "weight":
                        slice_vmin = 0
                        slice_vmax = np.nanquantile(map_slice_data[1:-1, 1:-1], q=0.99)

                    else:
                        subset = np.random.choice(map_slice_data.size, size=min(map_slice_data.size, 100000), replace=False)
                        slice_vmin, slice_vmax = np.nanquantile(
                            map_slice_data.ravel()[subset],
                            weights=map_slice_weight.ravel()[subset],
                            q=(rel_vmin, rel_vmax),
                            method="inverted_cdf",
                        )
                        if center_zero:
                            abs_max = max(slice_vmax, -slice_vmin)
                            slice_vmin, slice_vmax = -abs_max, abs_max

                    logger.debug(f"Inferring (vmin, vmax) = ({slice_vmin}, {slice_vmax})")

                ref = ax.pcolormesh(
                    getattr(self.xi, grid_hu["units"]),  # this is correct actually; the indexing is different here
                    getattr(self.eta, grid_hu["units"]),
                    map_slice_data,
                    cmap=cmap,
                    vmin=vmin or slice_vmin,
                    vmax=vmax or slice_vmax,
                    shading="nearest",
                )

                ax.set_aspect("equal")

                cbar = fig.colorbar(
                    ref,
                    ax=ax,
                    shrink=0.8,
                    aspect=16,
                    location="right",
                    pad=0.01,
                )

                slice_info = []

                if "stokes" in ax_slices:
                    slice_info.append(f"Stokes {self.stokes[ax_slices['stokes']]}")
                if "nu" in ax_slices:
                    slice_info.append(f"{self.nu[ax_slices['nu']]}")
                if "v" in ax_slices:
                    slice_info.append(f"{self.v[ax_slices['v']]}")
                if "t" in ax_slices:
                    slice_info.append(f"{self.t[ax_slices['t']]}")

                cbar.set_label(rf"${map_units_repr}$ ({', '.join(slice_info)})", fontsize=10)

                ax.tick_params(axis="x", bottom=True, top=False)
                ax.tick_params(axis="y", left=True, right=False, rotation=90)

                # lon = ax.coords[0]
                # lon.set_axislabel(rf"{self.frame.phi_long_name}")

                # lat = ax.coords[1]
                # lat.set_axislabel(rf"{self.frame.theta_long_name}")

                ax.set_xlabel(rf"{self.frame.phi_long_name}")
                ax.set_ylabel(rf"{self.frame.theta_long_name}")

                ax2 = ax.secondary_xaxis("top")
                ax2.set_xlabel(rf"$\Delta\,\theta$ [{grid_hu['units']}]")

        if filename is not None:
            plt.savefig(fname=filename, dpi=256)

    def to_fits(self, path: str):

        # if self.dims.get("nu", 0) > 1:
        #     logger.warning("Writing a multifrequency maps ")

        header = self.header()
        hdus = [fits.PrimaryHDU(data=np.asarray(self.data), header=header)]

        # the weight (if any) goes in a second image extension with the same WCS
        if self._weight is not None:
            hdus.append(fits.ImageHDU(data=np.asarray(self._weight), header=header, name="WEIGHT"))

        fits.HDUList(hdus).writeto(path, overwrite=True)

    def to_hdf(self, filename: str, compress: bool = True):
        compression_kwargs = {"compression": "gzip", "compression_opts": 9} if compress else {}

        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=np.float32, data=self.data, **compression_kwargs)
            if not (self.weight == 1).all().compute():
                f.create_dataset("weight", dtype=np.float32, data=self.weight, **compression_kwargs)
            if "stokes" in self.dims:
                f.create_dataset("stokes", data="".join(self.stokes), dtype=h5py.string_dtype())
            if "nu" in self.dims:
                f.create_dataset("nu", data=self.nu.Hz)
            if "t" in self.dims:
                f.create_dataset("t", data=self.t.s)
            if "v" in self.dims:
                f.create_dataset("v", data=self.v.to("m/s"))
            if "z" in self.dims:
                f.create_dataset("z", data=self.z)
            f.create_dataset("center", dtype=float, data=(self.center[0].deg, self.center[1].deg))
            f.create_dataset("eta", dtype=float, data=self.eta.deg)
            f.create_dataset("xi", dtype=float, data=self.xi.deg)
            f.create_dataset("units", data=self.units)
            f.create_dataset("frame", data=self.frame.name)
            f.create_dataset("beam", data=self.beam.deg)
