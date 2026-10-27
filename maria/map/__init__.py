from __future__ import annotations

import logging
import os
import re
import warnings

import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from matplotlib.colors import ListedColormap

from ..coords import frames
from ..io import fetch, read_fits_map
from ..units import Quantity, parse_units
from .base import VALID_MAP_QUANTITIES, Map, concatenate  # noqa
from .healpix_map import HEALPixMap  # noqa
from .projection_map import ProjectionMap  # noqa
from .transfer import TransferFunction, compute_transfer_function, plot_transfer_function  # noqa

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

with open(f"{here}/maps.txt", "r") as f:
    all_maps = f.read().splitlines()

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/../plotting/Planck_Parchment_RGB.txt") / 255.0,
    name="cmb",
)
cmb_cmap.set_bad("white")
mpl.colormaps.register(cmb_cmap)

MAP_SIZE_KWARGS = ["xi", "eta", "width", "height", "xi_res", "eta_res", "resolution"]
VALID_MAP_KWARGS = ["stokes", "nu", "t", "center", "frame", "units", "beam", *MAP_SIZE_KWARGS]

FITS_KEYWORD_ALIASES = {
    "frame": ["FRAME"],
    "units": ["BUNIT", "BUNITS"],
    "nu": ["FREQ", "RESTFREQ"],
}

AXIS_MAPPING = {
    "nu": {
        "aliases": ["FREQ", "NU"],
        "default_units": "Hz",
    },
    "v": {
        "aliases": ["VRAD", "VELO"],
        "default_units": "m/s",
    },
}


def get(name: str, **map_kwargs):
    return load(fetch(name), **map_kwargs)


def load(filename: str, index: int = None, format: str = "auto", **map_kwargs) -> Map:
    if format == "auto":
        format = filename.split(".")[-1].lower()

    if format == "fits":
        data, axis_mask, kwargs, header = read_fits_map(filename, index=index, strict=False)
        squeezed_axes = tuple([i for i in range(data.ndim) if not axis_mask[i]])
        data = np.squeeze(data, squeezed_axes)
        if "weight" in kwargs:
            kwargs["weight"] = np.squeeze(kwargs["weight"], squeezed_axes)
    elif format == "h5":
        data, kwargs = read_hdf_map(filename)
    else:
        raise NotImplementedError(f"Unsupported filetype '.{format}'.")

    # if there are any kwargs specifying the size of the map
    # remove size kwargs from the read-in map's metadata
    # so that they overwrite the size
    file_xi, file_eta = (kwargs.get("xi"), kwargs.get("eta")) if format == "fits" else (None, None)
    overridden_size_kwargs = {}
    overriding_size_kwargs = {k: v for k, v in map_kwargs.items() if k in MAP_SIZE_KWARGS}
    if overriding_size_kwargs:
        for k in MAP_SIZE_KWARGS:
            if k in kwargs:
                overridden_size_kwargs[k] = kwargs.pop(k)

        logging.info(
            f"Passed map size kwargs {overriding_size_kwargs} will overwrite map size metadata {overridden_size_kwargs}"
        )

    # a FITS map is placed by its reference pixel, which need not be in its middle; a center or a size passed here
    # applies to the middle of the map, as for a map without a reference pixel
    if file_xi is not None and file_eta is not None and ("center" in map_kwargs or overriding_size_kwargs):
        if "center" not in map_kwargs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                # only the two sky axes: a degenerate extra axis (CDELT = 0) makes the full WCS singular
                wcs = WCS(header, naxis=2)
            phi, theta = wcs.wcs_pix2world([[(file_xi.size - 1) / 2, (file_eta.size - 1) / 2]], 0)[0]
            kwargs["center"] = (float(phi), float(theta))
        if not overriding_size_kwargs:
            kwargs["xi"] = file_xi - (file_xi[0] + file_xi[-1]) / 2
            kwargs["eta"] = file_eta - (file_eta[0] + file_eta[-1]) / 2

    kwargs.update(map_kwargs)
    logger.debug(f"Loading ProjectionMap with kwargs {kwargs}")

    return ProjectionMap(data=data, **kwargs)


def read_hdf_map(filename: str):
    with h5py.File(filename, "r") as f:
        kwargs = {}
        for field in f.keys():
            value = f[field][()]
            kwargs[field] = value if not isinstance(value, bytes) else value.decode()

        if "weight" in f.keys():
            kwargs["weight"] = f["weight"][:]

    data = kwargs.pop("data")

    return data, kwargs
