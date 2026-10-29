from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import dask.array as da
import numpy as np
import scipy as sp
from jax import jit
from jax import scipy as jsp
from tqdm import tqdm

from ..beam import compute_angular_fwhm
from ..calibration import Calibration
from ..constants import k_B
from ..io import DEFAULT_BAR_FORMAT, DEFAULT_TIME_FORMAT, humanize_time
from ..map import HEALPixMap, Map, get
from ..units import Quantity

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

DEFAULT_MAP_SIM_KWARGS = {
    "bilinear_sampling": False,
}


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _initialize_map(self, map: str | Map, **map_kwargs):
        if isinstance(map, str):
            # the simulation settings are not map metadata
            m = get(map, **{k: v for k, v in map_kwargs.items() if k not in DEFAULT_MAP_SIM_KWARGS})
        elif isinstance(map, Map):
            invalid_kwargs = [key for key in map_kwargs if key not in DEFAULT_MAP_SIM_KWARGS]
            if invalid_kwargs:
                raise ValueError(
                    f"Invalid map_kwargs {invalid_kwargs}; valid ones are {list(DEFAULT_MAP_SIM_KWARGS.keys())}."
                )
            m = map.copy()
        else:
            raise ValueError("'map' must be either a Map or a string")

        # the map can be frequency-naive if it is already in K_RJ
        # self.map = map.to(units="K_RJ")

        if "stokes" not in m.dims:
            m = m.unsqueeze("stokes")

        if "nu" not in m.dims:
            m = m.unsqueeze("nu")

        if "t" in m.dims:
            if m.dims["t"] > 1:
                map_start = arrow.get(m.t.seconds.min()).to("utc")
                map_end = arrow.get(m.t.seconds.max()).to("utc")
                if map_start > self.min_time:
                    logger.warning(
                        f"Beginning of map ({map_start.format(DEFAULT_TIME_FORMAT)}) is after the "
                        f"beginning of the simulation ({self.min_time.format(DEFAULT_TIME_FORMAT)}).",
                    )
                if map_end < self.max_time:
                    logger.warning(
                        f"End of map ({map_end.format(DEFAULT_TIME_FORMAT)}) is before the "
                        f"end of the simulation ({self.max_time.format(DEFAULT_TIME_FORMAT)}).",
                    )
        else:
            m = m.unsqueeze("t")

        parity_signature = {dim: (-1 if dim in ["v", "eta"] else 1) for dim in m.dims}
        m.apply_parity(**parity_signature)

        self.maps["map"] = m

    def _run(self, **kwargs):
        self._sample_maps(**kwargs)

    def _sample_maps(self, obs):

        for map_name, m in self.maps.items():
            map_loading = np.zeros(obs.coords.shape, dtype=self.dtype)

            # the detectors of one band can sit behind different apertures, which give them different beams
            primary_sizes = np.asarray(obs.instrument.dets.primary_size)
            band_groups = []
            for band in obs.instrument.dets.bands:
                in_band = np.asarray(obs.instrument.dets.band_name == band.name)
                for primary_size in np.unique(primary_sizes[in_band]):
                    band_groups.append((band, in_band & (primary_sizes == primary_size), primary_size))

            bands_pbar = tqdm(
                band_groups,
                desc=f"Sampling source '{map_name}'",
                disable=self.disable_progress_bars,
                bar_format=DEFAULT_BAR_FORMAT,
                ncols=250,
                postfix={"band": "", "channel": "", "stokes": ""},
            )
            for band, band_mask, primary_size in bands_pbar:
                bands_pbar.set_postfix(band=band.name)

                band_coords = obs.coords[band_mask]
                band_dets = obs.instrument.dets[band_mask]

                band_fwhm = Quantity(
                    compute_angular_fwhm(
                        fwhm_0=primary_size,
                        z=np.inf,
                        nu=band.center.Hz,
                    ),
                    "rad",
                )

                # ideally we would do this for each nu bin, but that's slow

                if isinstance(m, HEALPixMap):
                    # full-sky smoothing is slow, so do it once per beam and map data for all observations (Q and U
                    # as a spin-2 field). Polarized galactic maps are rotated to ra/dec, so that Q and U of every map
                    # refer to the same (celestial) north.
                    key = (map_name, float(band_fwhm.rad))
                    # (the map is kept with its data key, so that a new map or an edit of its data is not missed)
                    cached_map, cached_data, smoothed_map = self._smoothed_healpix_maps.get(key, (None, None, None))
                    if cached_map is not m or cached_data != m._data_key():
                        polarized = "stokes" in m.dims and {"Q", "U"} <= set(m.stokes)
                        frame = "ra/dec" if polarized and m.frame.name == "galactic" else None
                        smoothed_map = m._smooth(fwhm=band_fwhm, pol=True, frame=frame)
                        self._smoothed_healpix_maps[key] = (m, m._data_key(), smoothed_map)
                else:
                    smoothed_map = m.smooth(fwhm=band_fwhm)

                logger.debug(f"Convolved map with beam width {band_fwhm} for band {band.name}")

                bands_pbar.set_postfix(band=band.name, message=f"Computing pointing matrix")

                pointing_s = ttime.monotonic()

                # there is no sky signal beyond the edges of the input map
                P = smoothed_map[:, 0]._stokes_weighted_pointing_matrix(
                    coords=band_coords,
                    dets=band_dets,
                    bilinear=self.map_kwargs.get("bilinear_sampling", DEFAULT_MAP_SIM_KWARGS["bilinear_sampling"]),
                    zero_outside=True,
                )
                logger.debug(
                    f"Computed pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}"
                )

                for channel_index, (nu_min, nu_max) in enumerate(m.nu_bin_bounds):
                    if (band.nu.Hz.max() < nu_min) or (nu_max < band.nu.Hz.min()):
                        continue

                    nu_min = max(band.nu.Hz.min(), nu_min)
                    nu_max = min(band.nu.Hz.max(), nu_max)

                    channel_s = ttime.monotonic()
                    channel_map = smoothed_map[:, channel_index]
                    channel_map_data = channel_map.data.compute()
                    channel_string = f"{(Quantity(nu_min, 'Hz'), Quantity(nu_max, 'Hz'))}"

                    bands_pbar.set_postfix(band=band.name, message=f"Sampling channel {channel_string}")

                    calibration_s = ttime.monotonic()

                    if hasattr(obs, "atmosphere"):
                        # channel edges inside the band are added to the grid, so that adjacent channels partition it
                        nu = obs.atmosphere.spectrum.side_nu.Hz
                        channel_edges = [x for x in (nu_min, nu_max) if band.nu.Hz.min() < x < band.nu.Hz.max()]
                        channel_nu = np.unique([*nu[(nu >= nu_min) & (nu <= nu_max)], *channel_edges])
                        channel_opacity = sp.interpolate.interp1d(nu, obs.atmosphere.spectrum.fields["opacity"], axis=-1)(
                            channel_nu
                        )

                        instrument_transmission = band.passband(channel_nu)

                        static_atmospheric_transmission = obs.atmosphere.spectrum.transmission(
                            nu=channel_nu,
                            pwv=obs.zenith_scaled_pwv[band_mask].mean().compute(),
                            elevation=obs.coords.el[band_mask].mean(),
                        )

                        static_transmission = instrument_transmission * static_atmospheric_transmission

                        f_grid = (instrument_transmission * np.exp(-channel_opacity) * static_transmission).sum(axis=-1) / (
                            static_transmission**2
                        ).sum(axis=-1)

                        xi = (
                            obs.atmosphere.weather.effective_temperature.K,
                            obs.zenith_scaled_pwv[band_mask].compute(),
                            1 / np.sin(obs.coords.el[band_mask]),
                        )

                        f = jsp.interpolate.RegularGridInterpolator(
                            points=obs.atmosphere.spectrum.points[:3], values=f_grid
                        )(xi)

                    else:
                        # integrate over the part of the band that falls in this channel
                        channel_nu = np.linspace(nu_min, nu_max, 100)
                        static_atmospheric_transmission = np.zeros_like(channel_nu)

                        static_transmission = band.passband(channel_nu)
                        f = 1

                    # computing the spectrum for every map pixel would be silly
                    # compute the power for the smallest and largest values, and then interpolate!
                    map_values = np.nanpercentile(
                        channel_map_data, weights=channel_map.weight.compute(), method="inverted_cdf", q=[0, 100]
                    )

                    # the calibration might need these to compute T_RJ; it takes scalar areas, so a map in units per
                    # beam needs a single beam per channel
                    beam_areas = np.unique(channel_map.beam_area.sr)
                    if beam_areas.size > 1 and channel_map.u["physical_quantity"] == "spectral_flux_density_per_beam":
                        raise ValueError(f"The beam of map channel {channel_string} varies along the other dimensions.")
                    map_calibration_kwargs = {
                        "pixel_area": float(channel_map.pixel_area.sr),
                        "beam_area": float(beam_areas[0]),
                    }

                    map_values_trj_spectrum = Calibration(f"{channel_map.units} -> K_RJ", **map_calibration_kwargs)(
                        map_values[:, None], nu=channel_nu
                    )

                    logger.debug(
                        f"Computed {channel_map.units} -> pW calibration for band {band.name}, channel {channel_string} in "
                        f"{humanize_time(ttime.monotonic() - calibration_s)}"
                    )

                    map_values_power = k_B * np.trapezoid(map_values_trj_spectrum * static_transmission, x=channel_nu)

                    channel_power_map = np.interp(channel_map_data, map_values, map_values_power)

                    pW = 1e12 * f * (P @ channel_power_map.ravel()).reshape(band_coords.shape)

                    if not np.isfinite(pW).all():
                        assert False

                    map_loading[band_mask] += pW

                    if logger.level == 10:
                        logger.debug(
                            f"Computed map load {pW.shape} ~{Quantity(pW.std(), 'pW'):<8} for band {band.name}, "
                            f"channel {channel_string} in {humanize_time(ttime.monotonic() - channel_s)}"
                        )

                    del pW

                if map_loading[band_mask].sum() == 0:
                    logger.warning(f"No load from map for band {band.name}")

            # the above samples the map instantaneously
            # so you get all the lower noise per sample benefits without the smearing
            # obviously we can't continuum sample, so we convolve the map loading with a triangular kernel
            map_loading = sp.ndimage.convolve1d(map_loading, weights=np.array([0.25, 0.5, 0.25]), axis=-1)

            obs.loading[map_name] = da.asarray(map_loading, dtype=self.dtype)
