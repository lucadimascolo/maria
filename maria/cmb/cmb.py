from __future__ import annotations

import os

import healpy as hp
import numpy as np
from astropy.io import fits

from ..constants import z_CMB
from ..io import fetch
from ..map import HEALPixMap

CMB_SPECTRUM_SOURCE_URL = (
    "https://github.com/thomaswmorris/maria-data/raw/master/cmb/spectra/"
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
)
CMB_SPECTRUM_CACHE_PATH = "/tmp/maria-data/cmb/spectrum.txt"
CMB_SPECTRUM_CACHE_MAX_AGE = 30 * 86400  # one month

CMB_MAP_SOURCE_URL = "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_OID=15001"
CMB_MAP_CACHE_PATH = "cmb/planck.fits"
CMB_MAP_CACHE_MAX_AGE = 30 * 86400  # one month


CMB_SOURCES = {
    "planck": {"spectrum": "cmb/spectra/planck.csv"},
    "camb": {"spectrum": "/Users/tom/maria/data/cmb/spectra/camb.csv"},
}


def _find_healpix_columns(path):
    """
    The (hdu, column) of the Stokes I, Q and U maps in a HEALPix FITS file, found by column name (the Planck CMB
    file has the temperature and the polarization in separate extensions), or else the first three columns of the
    first table.
    """
    prefixes = {"I": ("TEMPERATURE", "I"), "Q": ("Q_",), "U": ("U_",)}
    with fits.open(path) as hdul:
        tables = [(i, hdu.columns.names) for i, hdu in enumerate(hdul) if isinstance(hdu, fits.BinTableHDU)]

    columns = {}
    for hdu_index, names in tables:
        for column_index, name in enumerate(names):
            for stokes, stokes_prefixes in prefixes.items():
                if stokes not in columns and name.upper().startswith(stokes_prefixes):
                    columns[stokes] = (hdu_index, column_index)

    if len(columns) < 3:
        return [(tables[0][0], column_index) for column_index in range(3)]
    return [columns[stokes] for stokes in "IQU"]


def get_cmb(nside: int = None, **kwargs):
    """
    The Planck CMB map, in K_CMB and galactic coordinates, degraded to `nside` (by default, to the coarsest
    resolution of its I, Q and U maps). The map contains the solar dipole (about 3.4 mK).
    """
    cmb_path = fetch(url=CMB_MAP_SOURCE_URL, max_age=CMB_MAP_CACHE_MAX_AGE)

    maps = [
        hp.fitsfunc.read_map(cmb_path, hdu=hdu, field=column, dtype=np.float32)
        for hdu, column in _find_healpix_columns(cmb_path)
    ]

    nside = nside or min(hp.npix2nside(m.size) for m in maps)
    data = np.stack([m if hp.npix2nside(m.size) == nside else hp.ud_grade(m, nside_out=nside) for m in maps], axis=0)

    return HEALPixMap(
        data=data[:, None, None],
        stokes="IQU",
        units="K_CMB",
        frame="galactic",
        z=z_CMB,
    )
