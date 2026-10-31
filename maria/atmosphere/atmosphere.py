from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import numpy as np
import scipy as sp
from jax import scipy as jsp
from numpy.linalg import LinAlgError
from tqdm import tqdm

from ..functions import approximate_normalized_matern
from ..instrument import Instrument
from ..io import DEFAULT_BAR_FORMAT, humanize_time
from ..spectrum import AtmosphericSpectrum
from ..units import Quantity
from ..utils import compute_aligning_transform, get_rotation_matrix_2d
from ..weather import Weather
from .extrusion import generate_layers
from .process import AutoregressiveProcess

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

SUPPORTED_MODELS_LIST = ["2d", "3d"]


class Atmosphere:
    def __init__(
        self,
        model: str = "2d",
        timestamp: float = arrow.now().timestamp(),
        region: str = "princeton",
        altitude: float = None,
        kwargs: dict = {},
        weather_quantiles: dict = {},
        weather_source: str = "era5",
        pwv_rms_frac: float = 0.03,
        max_height: float = 5e3,
        timestep: float = None,
        interpolation_method: str = "linear",
        disable_progress_bars: bool = False,
        seasonal: bool = True,
        diurnal: bool = True,
    ):
        if model not in SUPPORTED_MODELS_LIST:
            raise ValueError(
                f"Invalid model '{model}'. Supported models are {SUPPORTED_MODELS_LIST}.",
            )

        self.kwargs = kwargs

        self.weather = Weather(
            time=timestamp,
            region=region,
            altitude=altitude,
            override=self.kwargs.get("weather", {}),
            quantiles=weather_quantiles,
            seasonal=seasonal,
            diurnal=diurnal,
            source=weather_source,
        )

        # the spectrum must be evaluated at the pressure level of the site, at the time of the observation
        self.spectrum = AtmosphericSpectrum.from_weather(self.weather)

        # atmosphere_kwargs["layers"]["pwv_rms_frac"], if given, takes precedence over the argument
        self.pwv_rms_frac = self.kwargs.get("layers", {}).get("pwv_rms_frac", pwv_rms_frac)

        self.angular = False

        self.model = model
        self.max_height = max_height
        self.timestep = timestep

        self.interpolation_method = interpolation_method

        self.disable_progress_bars = disable_progress_bars

        self._initialized = False

    def initialize(self, obs):
        """
        Simulate a realization of PWV.
        """

        # fail early (rather than deep inside the loading computation) when the weather is outside the spectrum tables
        for name, value, side, units in [
            ("PWV", self.weather.pwv.mm, self.spectrum.side_pwv.mm, "mm"),
            ("effective temperature", self.weather.effective_temperature.K, self.spectrum.side_effective_temperature.K, "K"),
        ]:
            if not side.min() <= value <= side.max():
                raise ValueError(
                    f"The {name} of the weather ({value:.02f} {units}) for region '{self.weather.region}' at "
                    f"{self.weather.local_time.format('YYYY-MM-DD HH:mm ZZ')} is outside the range of the atmospheric "
                    f"spectrum ({side.min():.02f} to {side.max():.02f} {units}). Try other weather quantiles or a "
                    f"weather override."
                )

        self.layers = generate_layers(
            instrument=obs.instrument,
            boresight=obs.boresight,
            weather=self.weather,
            site=obs.site,
            mode=self.model,
            angular=self.angular,
            **{**self.kwargs.get("layers", {}), "pwv_rms_frac": self.pwv_rms_frac},
        )

        if self.timestep is None:
            min_fwhm = obs.instrument.dets.angular_fwhm(z=self.max_height).min()
            max_wind = Quantity((self.layers.wind_speed / self.layers.h).values, "rad/s").max()
            self.timestep = max(1e-1, (min_fwhm / max_wind).s) if max_wind > 0 else np.inf
            # the atmosphere is upsampled with a cubic spline, which needs at least 4 samples: shorten the timestep only
            # for observations that are too short for that, so that every other one keeps its timestep
            duration = np.ptp(obs.boresight.t)
            if duration <= 3 * self.timestep:
                self.timestep = duration / 4

        # rotate the focal plane by the plan roll, as in Observation
        roll_matrix = get_rotation_matrix_2d(np.radians(obs.plan.roll)).T

        self.boresight = obs.boresight.downsample(timestep=self.timestep)
        self.coords = self.boresight.broadcast(
            obs.instrument.dets.offsets @ roll_matrix,
            frame="az/el",
        )

        # this is a smaller version of the sim coords
        outer_coords = self.boresight.broadcast(
            obs.instrument.dets.outer().offsets @ roll_matrix,
            frame="az/el",
        )

        self.processes = {}

        center = outer_coords.center(frame="az/el")

        for process_index in tqdm(
            sorted(np.unique(self.layers.process_index)),
            desc="Constructing atmosphere",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
        ):
            process_init_s = ttime.monotonic()

            in_process = self.layers.process_index == process_index
            process_layers = self.layers.loc[in_process]

            # all processes must have the same velocity! so we compute it here.
            # vx and vy must have shape (n_time,)

            if process_layers.angular.any():
                vx = (
                    process_layers.wind_east.values[:, None] * np.cos(self.boresight.az)
                    - process_layers.wind_north.values[:, None] * np.sin(self.boresight.az)
                ) / process_layers.h.values[:, None]
                vy = (
                    -process_layers.wind_north.values[:, None] * np.cos(self.boresight.az)
                    - process_layers.wind_east.values[:, None] * np.sin(self.boresight.az)
                ) / (process_layers.h.values[:, None] * np.sin(self.boresight.el))
            else:
                # the process frame is that of Coordinates.project(), where x points north and y points east
                vx = process_layers.wind_north.values[:, None] * np.ones(
                    self.boresight.shape[-1],
                )
                vy = process_layers.wind_east.values[:, None] * np.ones(
                    self.boresight.shape[-1],
                )

            # weight each layer by its (non-negative) column of water vapor, so that the process wind is a mean of the
            # layer winds; in 2d there is one layer per process, and the weights cancel
            w = process_layers.total_water.values
            vx = (w[:, None] * vx).sum(axis=0) / w.sum()
            vy = (w[:, None] * vy).sum(axis=0) / w.sum()
            vz = np.zeros(vx.shape)

            if len(process_layers) > 1:
                process_layers = process_layers.loc[process_layers.index[[0, -1]]]

            # here we want enough points for a hull
            # we have made sure that any hull for this subpointing is a cover for the whole pointing

            process_points_for_hull_list = []

            for i, (layer_index, layer) in enumerate(process_layers.iterrows()):
                if layer.angular:
                    layer_x, layer_y = outer_coords.offsets(
                        center=center,
                        frame="az/el",
                    )

                else:
                    # the screen is static in the process frame, so for it to drift downwind (along +v)
                    # the sampling points must move upwind (along -v)
                    p = outer_coords.project(z=layer.h)
                    p -= np.cumsum(self.timestep * np.c_[vx, vy, vz][None], axis=-2)

                    process_points_for_hull_list.append(p[None])

            process_points_for_hull = np.concatenate(
                process_points_for_hull_list,
                axis=0,
            ).reshape(-1, 3)
            process_points_for_hull[..., 2] += 1e-6 * np.random.standard_normal(
                process_points_for_hull[..., 2].shape,
            )
            # process_points_for_hull = process_points_for_hull.reshape(-1, 3)

            process_init_s = ttime.monotonic()

            transform = compute_aligning_transform(
                process_points_for_hull,
                signature=(True, True, False),
            )
            tp = process_points_for_hull @ transform
            triangulation = sp.spatial.Delaunay(tp[..., 1:])

            logger.debug(f"Computed process bounds in {humanize_time(ttime.monotonic() - process_init_s)}.")

            min_tx, min_ty, min_tz = tp.min(axis=0)
            max_tx, max_ty, max_tz = tp.max(axis=0)

            cross_section_points_list = []
            layer_labels = []

            for i, (layer_index, layer) in enumerate(
                self.layers.loc[in_process].iterrows(),
            ):
                layer_min_ty, layer_max_ty = min_ty, max_ty

                if self.model == "3d":
                    # in 3d the hull is a strip that is tilted in (ty, tz), so each sublayer only needs the part of
                    # the cross-section that is inside the hull at its height (and not the full width of the hull)
                    dense_ty = np.linspace(min_ty, max_ty, 4096)
                    query_tz = np.clip(layer.h, min_tz + 1e-3, max_tz - 1e-3)
                    interior = triangulation.find_simplex(np.c_[dense_ty, np.full_like(dense_ty, query_tz)]) > -1
                    if interior.any():
                        dense_step = dense_ty[1] - dense_ty[0]
                        layer_min_ty = max(min_ty, dense_ty[interior].min() - dense_step)
                        layer_max_ty = min(max_ty, dense_ty[interior].max() + dense_step)

                n_cross_section = int(np.maximum(2, (layer_max_ty - layer_min_ty + 2 * layer.res) / layer.res))
                cross_section_points = np.stack(
                    [
                        np.linspace(layer_min_ty - layer.res, layer_max_ty + layer.res, n_cross_section),
                        layer.h * np.ones(n_cross_section),
                    ]
                ).T

                layer_labels.extend(n_cross_section * [layer_index])
                cross_section_points_list.append(cross_section_points)

                # res = layer.res
                # wide_lp_x_dense = np.linspace(min_ty, max_ty, 10000)
                # wide_lp_z_dense = layer.h * np.ones(len(wide_lp_x_dense))
                # wide_lp_dense = np.c_[wide_lp_x_dense, wide_lp_z_dense]
                # interior = triangulation.find_simplex(wide_lp_dense) > -1
                # # interior = interior if interior.any() else np.ones_like(interior, dtype=bool)
                # lp_dense_x = wide_lp_x_dense[interior]
                # lp_x_min = lp_dense_x.min() - 2 * res
                # lp_x_max = lp_dense_x.max() + 2 * res
                # n_lp = np.maximum(3, int((lp_x_max - lp_x_min) / res))
                # lp_x = np.linspace(lp_x_min, lp_x_max, n_lp)
                # lp_z = layer.h * np.ones(len(lp_x))
                # lp = np.c_[lp_x, lp_z]
                # n_lp = len(lp)

                # layer_labels.extend(n_lp * [layer_index])
                # cross_section_points_list.append(lp)

            cross_section_points = np.concatenate(cross_section_points_list, axis=0)

            extrusion_points = np.arange(
                min_tx - 2 * self.layers.res.min(),
                max_tx + 2 * self.layers.res.min(),
                self.layers.res.min(),
            )

            outer_scale = np.maximum(1e3, 300 + process_layers.h.mean() / 10)

            matern_kwargs = {"nu": 1 / 3, "r0": outer_scale} if self.model == "3d" else {"nu": 5 / 6, "r0": outer_scale}

            process = AutoregressiveProcess(
                cross_section=cross_section_points,
                extrusion=extrusion_points,
                callback=approximate_normalized_matern,
                callback_kwargs=matern_kwargs,
                jitter=1e-8,
            )

            process.layers = self.layers.loc[in_process]
            process.labels = np.array(layer_labels)
            process.transform = transform
            process.extrusion_res = np.gradient(extrusion_points).mean()
            process.vx = vx
            process.vy = vy
            process.tp = tp

            # the 3d sublayers are one correlated field, so summing them also sums their covariances: rescale them so
            # that the PWV variance along the line of sight is the sum of the layer variances (as in 2d)
            process.amplitude_correction = 1.0
            if len(process.layers) > 1:
                a = process.layers.pwv_rms.values
                h = process.layers.h.values
                los_distance = np.abs(h[:, None] - h[None, :]) / np.sin(self.boresight.el.mean())
                C = approximate_normalized_matern(los_distance.ravel(), **matern_kwargs).reshape(los_distance.shape)
                if a @ C @ a > 0:  # all the sublayers have zero amplitude when pwv_rms_frac=0
                    process.amplitude_correction = float(np.sqrt((a @ a) / (a @ C @ a)))

            success = False
            for jitter in [1e-8, 1e-6, 1e-4]:
                process.jitter = jitter
                try:
                    process.compute_covariance_matrices()
                    success = True
                    break
                except LinAlgError as e:
                    logger.debug(f"Singular covariance matrix with jitter={jitter}")

            if not success:
                raise LinAlgError("Covariance matrix was singular even with max jitter.")

            self.processes[int(process_index)] = process

        self._initialized = True

    def __repr__(self):
        return f"""Atmosphere({len(self.processes)} processes with {len(self.layers)} layers):
├ spectrum:
│   region: {self.spectrum.region}
└ weather:
    region: {self.weather.region}
    altitude: {self.weather.altitude}
    time: {self.weather.local_time.format("MMM D HH:mm:ss ZZ")}
    pwv[mean, rms]: ({self.weather.pwv}, {self.pwv_rms_frac * self.weather.pwv})"""

    def simulate_pwv(self, instrument: Instrument = None):
        if not self._initialized:
            raise RuntimeError("Atmosphere must be initialized with a simulation.")

        pp = self.coords.project(z=1)

        for k, process in tqdm(
            self.processes.items(),
            desc="Generating turbulence",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
        ):
            process_s = ttime.monotonic()
            process.run()
            logger.debug(f"Ran atmospheric process {k} in {humanize_time(ttime.monotonic() - process_s)}")

        self.zenith_scaled_pwv = self.weather.pwv.mm * np.ones(shape=pp.shape[:-1])

        with tqdm(
            total=len(self.layers),
            desc="Sampling turbulence",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
        ) as pbar:
            for k, process in self.processes.items():
                wind_vector = np.c_[process.vx, process.vy, np.zeros(process.vx.shape)]
                # sampling points move upwind, so that the screen drifts downwind (see initialize)
                translation = -np.cumsum(self.timestep * wind_vector[None], axis=-2)

                for i, (layer_index, layer) in enumerate(
                    process.layers.iterrows(),
                ):
                    layer_s = ttime.monotonic()

                    layer_mask = process.labels == layer_index

                    p = layer.h * pp + translation
                    transformed_lpp = p @ process.transform

                    # sample each group of detectors with the same beam (e.g. each band) from the layer smoothed with it
                    det_fwhm = instrument.dets.physical_fwhm(layer.z).m if instrument else np.zeros(pp.shape[0])
                    rounded_fwhm = np.round(det_fwhm, 9)

                    for fwhm in np.unique(rounded_fwhm):
                        group_mask = rounded_fwhm == fwhm

                        beam_sigma = det_fwhm[group_mask].mean() / 2.355
                        extrusion_sigma = beam_sigma / process.extrusion_res
                        x_sigma = beam_sigma / layer.res

                        smoothed_layer_pwv = sp.ndimage.gaussian_filter(
                            process.values[:, layer_mask],
                            sigma=(extrusion_sigma, x_sigma),
                        )

                        y = jsp.interpolate.RegularGridInterpolator(
                            (
                                process.extrusion,
                                process.cross_section[layer_mask, 0],
                            ),
                            smoothed_layer_pwv,
                            method="linear",
                        )(transformed_lpp[group_mask][..., :2])

                        if np.isnan(y).any():
                            raise RuntimeError(f"Layer {layer_index} in process {k} introduced nans into PWV simulation.")

                        self.zenith_scaled_pwv[group_mask] += process.amplitude_correction * layer.pwv_rms * y

                    pbar.update(1)

                    logger.debug(
                        f"Sampled layer {layer_index} (h={Quantity(layer.h, 'm')}) with shape "
                        f"{smoothed_layer_pwv.shape} in {humanize_time(ttime.monotonic() - layer_s)}"
                    )
