from __future__ import annotations

import logging
import os
import re
import warnings

import numpy as np
from astropy.io import fits
from astropy.io.fits.header import Header
from astropy.wcs import WCS, FITSFixedWarning

from ...units import Quantity, parse_units

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")
FITS_TYPE_ALIASES = {
    "stokes": ["STOKES"],
    "nu": ["NU", "FREQ"],
    "v": ["VRAD", "VELO"],
    "t": ["TIME"],
    "z": ["REDSHIFT"],
}

FITS_KWARG_ALIASES = {
    "units": ["UNIT", "BUNIT", "BUNITS", "OUTTYPE"],
    "nu": ["NU", "FREQ", "RESTFRQ", "RESTFREQ"],
    "z": ["REDSHIFT"],
}

FITS_DEFAULT_UNITS = {
    "stokes": "",
    "nu": "Hz",
    "v": "m/s",
    "z": "",
    "t": "s",
    "eta": "deg",
    "xi": "deg",
}

FITS_FRAMES = {
    "ra/dec": {
        "xi": {"aliases": [r"^RA-*"], "parity": -1},
        "eta": {"aliases": [r"^DEC-*"], "parity": +1},
    },
    "galactic": {
        "xi": {"aliases": [r"^GLON-*"], "parity": -1},
        "eta": {"aliases": [r"^GLAT-*"], "parity": +1},
    },
}


def read_fits_map(path: str, index: int | None = None, strict: bool = True):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with fits.open(path) as hdul:
        if index is None:
            indices_with_image = [index for index, h in enumerate(hdul) if h.data is not None]
            if len(indices_with_image) == 0:
                raise ValueError(f"Could not infer HDU index (FITS file '{path}' has no images).")
            index = indices_with_image[0]

        hdu = hdul[index]

        # FITS counts from the bottom, while normal people count from the top
        data = hdu.data  # [..., ::-1, :]
        header = hdu.header

        # maria writes the map weight in an extension named WEIGHT
        weight = None
        if "WEIGHT" in hdul and hdul["WEIGHT"].data is not None and hdul["WEIGHT"].data.shape == data.shape:
            weight = hdul["WEIGHT"].data

    try:
        kwargs, axis_mask = parse_fits_map_header(header)
    except Exception as e:
        if strict:
            raise e
        logger.warning("Could not parse FITS header!")
        kwargs = {}
        axis_mask = np.ones(data.ndim, dtype=bool)

    if weight is not None:
        kwargs["weight"] = weight

    return data, axis_mask, kwargs, header


def _compute_fits_map_center(header, xi_axis: int, eta_axis: int, default):
    """
    The sky position of the centre of the image (0-based pixel ((NAXIS - 1) / 2) on each spatial axis).
    This is used for projections other than maria's own (ARC), where CRVAL can be far from the map.
    """
    pixel_center = [(header[f"NAXIS{axis}"] - 1) / 2 for axis in (xi_axis, eta_axis)]
    reference_pixel = [header.get(f"CRPIX{axis}", 1.0) - 1 for axis in (xi_axis, eta_axis)]

    if np.allclose(pixel_center, reference_pixel):
        return default

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            phi, theta = WCS(header, naxis=[xi_axis, eta_axis]).wcs_pix2world([pixel_center], 0)[0]
    except Exception as e:
        logger.warning(f"Could not compute the map center from the WCS ({e}), using CRVAL.")
        return default

    return [float(phi), float(theta)]


def parse_fits_map_header(header):

    parity = {}
    axes = {}
    kwargs = {}

    axis_mask = np.zeros(header["NAXIS"], dtype=bool)

    center = [0, 0]
    spatial_axes = {}
    spatial_offsets = {}
    for axis in range(header["NAXIS"]):
        parity = 1

        AXIS = axis + 1

        NAXIS = header[f"NAXIS{AXIS}"]
        CTYPE = header.get(f"CTYPE{AXIS}")

        if CTYPE is None:
            continue

        dim = None
        for dim_name, dim_aliases in FITS_TYPE_ALIASES.items():
            for alias in dim_aliases:
                if re.compile(alias).match(CTYPE):
                    dim, default_units = dim_name, FITS_DEFAULT_UNITS[dim_name]
                    break

        for frame_name, frame_config in FITS_FRAMES.items():
            for dim_name, dim_config in frame_config.items():
                for alias in dim_config["aliases"]:
                    if re.compile(alias).match(CTYPE):
                        dim, default_units = dim_name, FITS_DEFAULT_UNITS[dim_name]
                        kwargs["frame"] = frame_name
                        parity = dim_config["parity"]
                        break

        if dim is None:
            if NAXIS == 1:
                logger.warning(f"Squashing AXIS{AXIS} (could not interpret CTYPE{AXIS} = {repr(CTYPE)})")
                continue
            else:
                raise ValueError(f"Could not interpret CTYPE{AXIS} = {repr(CTYPE)}")
        else:
            axis_mask[-AXIS] = True

        CDELT = header.get(f"CDELT{AXIS}")
        CRVAL = header.get(f"CRVAL{AXIS}")
        CRPIX = header.get(f"CRPIX{AXIS}", 1.0)
        CUNIT = header.get(f"CUNIT{AXIS}", default_units)

        logger.debug(f"Found CTYPE{AXIS}: {repr(CTYPE)}")
        logger.debug(f"Found CDELT{AXIS}: {CDELT}")
        logger.debug(f"Found CRVAL{AXIS}: {CRVAL}")
        logger.debug(f"Found CRPIX{AXIS}: {CRPIX}")
        logger.debug(f"Found CUNIT{AXIS}: {repr(CUNIT)}")
        logger.debug(f"Interpreting axis {AXIS} as dimension '{dim}'")

        CDELT *= parity
        logger.debug(f"Applying parity {['-1', '+1'][int(parity > 0)]} for dimension '{dim}'")

        if dim in ["xi", "eta"]:
            kwargs[f"{dim}_res"] = CDELT
            # the offsets from CRVAL, which is at the (1-based) reference pixel CRPIX
            spatial_offsets[dim] = CDELT * (np.arange(NAXIS, dtype=float) - (CRPIX - 1))
            center[["xi", "eta"].index(dim)] = CRVAL
            spatial_axes[dim] = AXIS
        else:
            # maria used to write CRPIX = 0 with CRVAL at the first pixel, and 0-based Stokes indices
            legacy = CRPIX == 0
            # CRPIX is 1-based
            axis_values = CRVAL + CDELT * (np.arange(NAXIS, dtype=float) - (0 if legacy else CRPIX - 1))
            if dim == "stokes":
                # FITS Stokes codes (I = 1, Q = 2, U = 3, V = 4) to maria's 0-based indices into "IQUV"
                axis_values = np.round(axis_values).astype(int) - (0 if legacy else 1)
            axes[dim] = Quantity(axis_values, CUNIT)

    # CRVAL is the sky position of the reference pixel CRPIX, which need not be the middle of the image
    if len(spatial_axes) == 2:
        projections = {str(header[f"CTYPE{AXIS}"])[5:8] for AXIS in spatial_axes.values()}
        if projections == {"ARC"}:
            # maria's projection (ARC) about CRVAL, with the offsets counted from CRPIX, is exact wherever CRPIX is
            for dim in ["xi", "eta"]:
                kwargs[dim] = spatial_offsets[dim]
                kwargs.pop(f"{dim}_res")
        else:
            # other projections (TAN, SIN, CAR, ...) differ from ARC away from CRVAL, which can be far from the map:
            # center the map on the sky position of its middle pixel
            center = _compute_fits_map_center(header, spatial_axes["xi"], spatial_axes["eta"], default=center)

    # a keyword (like RESTFRQ, the rest frequency of a line) does not override an axis of the data, even a
    # single-channel one: a FREQ axis of size 1 gives the frequency of the map, and RESTFRQ is then ignored
    data_axes = set(axes)
    for key, value in header.items():
        for kwarg, aliases in FITS_KWARG_ALIASES.items():
            for alias in aliases:
                if key == alias:
                    logger.debug(f"Using {alias}: {value} for kwarg '{kwarg}'")
                    if kwarg in ["nu", "z"]:
                        if kwarg not in data_axes:
                            axes[kwarg] = Quantity([value], FITS_DEFAULT_UNITS[kwarg])
                    else:
                        kwargs[kwarg] = value
                    break

    beam = [0, 0, 0]
    beam_keys = [["BMAJ", "BMAJOR"], ["BMIN", "BMINOR"], ["BPA"]]
    for beam_index, aliases in enumerate(beam_keys):
        for alias in aliases:
            if alias in header:
                beam[beam_index] = header[alias]
                logger.debug(f"Using {alias}: {value} as beam parameter")
                break

    return {"center": center, "beam": beam, **kwargs, **axes}, axis_mask
