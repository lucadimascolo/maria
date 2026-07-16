from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import numpy as np
from tqdm import tqdm

from ..atmosphere import Atmosphere
from ..cmb import generate_cmb, get_cmb
from ..coords import Coordinates
from ..errors import PointingError
from ..instrument import Instrument, get_instrument
from ..io import DEFAULT_BAR_FORMAT, fetch, humanize_time, read_yaml
from ..map import Map, load
from ..plan import Plan, PlanList, get_plan
from ..site import Site, get_site
from ..tod import TOD
from .atmosphere import DEFAULT_ATMOSPHERE_SIM_KWARGS, AtmosphereMixin  # noqa
from .base import BaseSimulation
from .cmb import DEFAULT_CMB_SIM_KWARGS, CMBMixin  # noqa
from .map import DEFAULT_MAP_SIM_KWARGS, MapMixin  # noqa
from .noise import DEFAULT_NOISE_SIM_KWARGS, NoiseMixin  # noqa
from .observation import Observation

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

MIN_ELEVATION_WARN = 10  # degrees
MIN_ELEVATION_ERROR = 5  # degrees


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!",
        )


master_params = read_yaml(f"{here}/params.yml")


def parse_sim_kwargs(kwargs, master_kwargs, strict=False):
    parsed_kwargs = {k: {} for k in master_kwargs.keys()}
    invalid_kwargs = {}

    for k, v in kwargs.items():
        parsed = False
        for sub_type, sub_kwargs in master_kwargs.items():
            if k in sub_kwargs.keys():
                parsed_kwargs[sub_type][k] = v
                parsed = True
        if not parsed:
            invalid_kwargs[k] = v

    if len(invalid_kwargs) > 0:
        if strict:
            raise InvalidSimulationParameterError(
                invalid_keys=list(invalid_kwargs.keys()),
            )

    return parsed_kwargs


class Simulation(AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """
    A simulation of a telescope. This is what users should touch, primarily.
    """

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**{**config, **params})

    def __init__(
        self,
        instrument: Instrument | str,
        plans: PlanList | list[Plan | str],
        site: Site | str,
        atmosphere: Atmosphere | str = None,
        atmosphere_kwargs: dict = {},
        cmb: CMB | str = None,
        cmb_kwargs: dict = {},
        map: Map | str = None,
        map_kwargs: dict = {},
        noise: bool = True,
        noise_kwargs: dict = {},
        progress_bars: bool = True,
        keep_mean_signal: bool = False,
        dtype: type = np.float32,
        seed: int = None,
        n_chunks: int = None,
    ):
        self.atmosphere = atmosphere
        self.atmosphere_kwargs = DEFAULT_ATMOSPHERE_SIM_KWARGS.copy()
        self.atmosphere_kwargs.update(atmosphere_kwargs)

        self.map_kwargs = map_kwargs

        self.noise = noise
        self.noise_kwargs = noise_kwargs

        self.dtype = dtype

        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self._mpi_comm = None
        self._mpi_rank = 0
        self._mpi_size = 1
        try:
            from mpi4py import MPI

            self._mpi_comm = MPI.COMM_WORLD
            self._mpi_rank = self._mpi_comm.Get_rank()
            self._mpi_size = self._mpi_comm.Get_size()
        except ImportError:
            pass

        self._mpi = self._mpi_size > 1
        self._n_chunks = n_chunks or 1
        self.disable_progress_bars = (
            (not progress_bars)
            or (logging.getLevelName(logger.level) == "DEBUG")
            or (self._mpi_rank != 0)
        )

        instrument_init_s = ttime.monotonic()

        # parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if isinstance(instrument, Instrument):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(name=instrument)

        logger.debug(f"Initialized instrument in {humanize_time(ttime.monotonic() - instrument_init_s)}.")
        site_init_s = ttime.monotonic()

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site)
        else:
            raise ValueError(
                "'site' must be either a Site object or a string.",
            )

        logger.debug(f"Initialized site in {humanize_time(ttime.monotonic() - site_init_s)}.")
        plan_init_s = ttime.monotonic()

        if isinstance(plans, str):
            plans = [get_plan(plan_name=plans)]
        elif isinstance(plans, Plan):
            plans = [plans]
        elif isinstance(plans, PlanList):
            plans = plans
        elif not isinstance(plans, list):
            raise TypeError("plans must be a plan or a list of plans")

        self.plans = PlanList(plans)

        logger.debug(f"Initialized plans in {humanize_time(ttime.monotonic() - plan_init_s)}.")

        self._base_instrument = self.instrument

        if self._mpi:
            n_dets = len(self.instrument.dets)
            all_indices = np.arange(n_dets)
            # Interleaved (stride) assignment ensures every rank gets a
            # proportional mix of all frequency bands even when detectors are
            # band-ordered. Contiguous blocks would give each rank only a
            # single band, causing unequal nu grids and atmosphere hull sizes.
            rank_indices = all_indices[self._mpi_rank :: self._mpi_size]
            mask = np.zeros(n_dets, dtype=bool)
            mask[rank_indices] = True
            self.instrument = self.instrument._subset(mask)
            logger.info(
                f"MPI rank {self._mpi_rank}/{self._mpi_size}: "
                f"assigned {len(rank_indices)} of {n_dets} detectors (interleaved)"
            )

        self.obs_list = self._build_obs_list(self.instrument)
        self.maps = {}
        if cmb:
            cmb_start_s = ttime.monotonic()
            self.cmb_kwargs = DEFAULT_CMB_SIM_KWARGS.copy()
            self.cmb_kwargs.update(cmb_kwargs)
            self._init_cmb(cmb, **cmb_kwargs)
            logger.debug(f"Initialized CMB simulation in {humanize_time(ttime.monotonic() - cmb_start_s)}.")

        if map:
            map_start_s = ttime.monotonic()
            self.map_kwargs = DEFAULT_MAP_SIM_KWARGS.copy()
            self.map_kwargs.update(map_kwargs)
            self._initialize_map(map, **map_kwargs)
            logger.debug(f"Initialized map simulation in {humanize_time(ttime.monotonic() - map_start_s)}.")

        if noise:
            noise_start_s = ttime.monotonic()
            self.noise_kwargs = DEFAULT_NOISE_SIM_KWARGS.copy()
            self.noise_kwargs.update(noise_kwargs)
            logger.debug(f"Initialized noise simulation in {humanize_time(ttime.monotonic() - noise_start_s)}.")

        #     # self.start = arrow.get(self.boresight.t.min()).to("utc")
        #     # self.end = arrow.get(self.boresight.t.max()).to("utc")

        #     if atmosphere:
        #         atmosphere_init_start_s = ttime.monotonic()

        #         # give it the observation, so that it knows about pointing, site, etc. (kind of cursed)
        #         obs.atmosphere.initialize(obs)

        #         logger.debug(
        #             f"Initialized atmosphere simulation in {humanize_time(ttime.monotonic() - atmosphere_init_start_s)}."
        #         )

        # logger.debug(f"Initialized simulation in {humanize_time(ttime.monotonic() - sim_start_s)}.")

    def _build_obs_list(self, instrument):
        obs_list = []
        for obs_index, plan in enumerate(self.plans):
            logger.debug(f"Initializing Observation {obs_index + 1} of {len(self.plans)}")
            obs_start_s = ttime.monotonic()
            obs = Observation(
                instrument=instrument,
                plan=plan,
                site=self.site,
                atmosphere=self.atmosphere,
                atmosphere_kwargs=self.atmosphere_kwargs,
            )
            if hasattr(obs, "atmosphere"):
                obs.atmosphere.initialize(obs, reference_instrument=self._base_instrument)
            obs_list.append(obs)
            logger.debug(f"Initialized Observation in {humanize_time(ttime.monotonic() - obs_start_s)}.")
        return obs_list

    def _build_obs_list_from_reference(self, ref_obs_list, instrument):
        """Build obs list reusing already-initialized atmosphere from ref_obs_list.

        The atmosphere spatial field (AR process, covariance matrices, GP grid) is
        identical for every chunk, so we only construct it once. Here we reuse that
        field and update only the per-detector sampling coordinates.
        """
        obs_list = []
        for ref_obs in ref_obs_list:
            obs = Observation(
                instrument=instrument,
                plan=ref_obs.plan,
                site=self.site,
                atmosphere=None,
                atmosphere_kwargs=self.atmosphere_kwargs,
            )
            if hasattr(ref_obs, "atmosphere"):
                obs.atmosphere = ref_obs.atmosphere
                obs.atmosphere.coords = ref_obs.atmosphere.boresight.broadcast(
                    instrument.dets.offsets,
                    frame="az/el",
                )
            obs_list.append(obs)
        return obs_list

    def run(self, units: str = "K_RJ", save_dir: str = None):
        # Reseed here so that the atmosphere AR process always starts from the
        # same PRNG state on every rank, regardless of how many random values
        # were consumed during __init__ (which varies with n_dets per rank).
        if self._seed is not None:
            np.random.seed(self._seed)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        n_dets = len(self.instrument.dets)
        all_indices = np.arange(n_dets)
        all_tods, saved_paths = [], []
        ref_obs_list = None

        for chunk_idx in range(self._n_chunks):
            if self._n_chunks > 1:
                mask = np.zeros(n_dets, dtype=bool)
                mask[all_indices[chunk_idx :: self._n_chunks]] = True
                chunk_instrument = self.instrument._subset(mask)
                logger.info(
                    f"Chunk {chunk_idx + 1}/{self._n_chunks}: "
                    f"{mask.sum()} of {n_dets} detectors (interleaved)"
                )
                if self._seed is not None:
                    np.random.seed(self._seed)
            else:
                chunk_instrument = self.instrument

            if ref_obs_list is None:
                obs_list = self._build_obs_list(chunk_instrument)
                ref_obs_list = obs_list
            else:
                obs_list = self._build_obs_list_from_reference(ref_obs_list, chunk_instrument)

            for obs_index, obs in enumerate(obs_list):
                logger.info(f"Simulating observation {obs_index + 1} of {len(obs_list)}")
                obs_start_s = ttime.monotonic()
                tod = self.run_obs(obs).to(units)
                logger.info(
                    f"Simulated observation {obs_index + 1} of {len(obs_list)} "
                    f"in {humanize_time(ttime.monotonic() - obs_start_s)}"
                )
                if save_dir:
                    obs_suffix = f"_obs{obs_index:03d}" if len(self.plans) > 1 else ""
                    path = os.path.join(save_dir, f"tod_rank-{self._mpi_rank}_chunk-{chunk_idx}{obs_suffix}.h5")
                    tod.save(path)
                    saved_paths.append(path)
                    logger.info(f"Saved → {path}")
                else:
                    all_tods.append(tod)

        return saved_paths if save_dir else all_tods

    def stream(self, units: str = "K_RJ"):
        """Yield one TOD at a time, sharing the atmospheric realization across chunks.

        Intended for memory-efficient streaming pipelines where the caller
        accumulates each TOD (e.g. into a mapper) and discards it immediately.
        The atmosphere GP field is retained on ``ref_obs_list`` between chunks,
        so all chunks see the same atmospheric realization.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        n_dets = len(self.instrument.dets)
        all_indices = np.arange(n_dets)
        ref_obs_list = None

        for chunk_idx in range(self._n_chunks):
            if self._n_chunks > 1:
                mask = np.zeros(n_dets, dtype=bool)
                mask[all_indices[chunk_idx :: self._n_chunks]] = True
                chunk_instrument = self.instrument._subset(mask)
                logger.info(
                    f"Chunk {chunk_idx + 1}/{self._n_chunks}: "
                    f"{mask.sum()} of {n_dets} detectors (interleaved)"
                )
                if self._seed is not None:
                    np.random.seed(self._seed)
            else:
                chunk_instrument = self.instrument

            if ref_obs_list is None:
                obs_list = self._build_obs_list(chunk_instrument)
                ref_obs_list = obs_list
            else:
                obs_list = self._build_obs_list_from_reference(ref_obs_list, chunk_instrument)

            for obs_index, obs in enumerate(obs_list):
                logger.info(f"Simulating observation {obs_index + 1} of {len(obs_list)}")
                obs_start_s = ttime.monotonic()
                tod = self.run_obs(obs).to(units)
                logger.info(
                    f"Simulated observation {obs_index + 1} of {len(obs_list)} "
                    f"in {humanize_time(ttime.monotonic() - obs_start_s)}"
                )
                yield tod

    def run_obs(self, obs: Observation) -> TOD:
        obs.loading = {}

        if hasattr(obs, "atmosphere"):
            atmosphere_sim_start_s = ttime.monotonic()
            # obs.atmosphere.initialize(obs)
            self._simulate_atmosphere(obs)
            obs.loading["atmosphere"] = self._compute_atmospheric_loading(obs)
            logger.debug(f"Ran atmosphere simulation in {humanize_time(ttime.monotonic() - atmosphere_sim_start_s)}.")

        # if hasattr(self, "cmb"):
        #     cmb_sim_start_s = ttime.monotonic()
        #     obs.loading["cmb"] = self._compute_cmb_loading(obs)
        #     logger.debug(f"Ran CMB simulation in {humanize_time(ttime.monotonic() - cmb_sim_start_s)}.")

        if self.maps:
            map_sim_start_s = ttime.monotonic()
            self._sample_maps(obs)
            logger.debug(f"Ran map simulation in {humanize_time(ttime.monotonic() - map_sim_start_s)}.")

        # number of bands are lost here
        if self.noise:
            noise_sim_start_s = ttime.monotonic()
            self._simulate_noise(obs)
            logger.debug(f"Ran noise simulation in {humanize_time(ttime.monotonic() - noise_sim_start_s)}.")

        if self._seed is not None:
            gain_values = np.array([
                np.random.default_rng([self._seed + 1, int(g)]).standard_normal()
                for g in obs.instrument._global_det_indices
            ])
        else:
            gain_values = np.random.standard_normal(size=obs.instrument.dets.n)
        gain_error = np.exp(obs.instrument.dets.gain_error * gain_values)

        for field in obs.loading:
            if field in ["noise"]:
                continue

            obs.loading[field] *= gain_error[:, None]

        metadata = {
            "atmosphere": False,
            "sim_time": arrow.now(),
            "altitude": float(obs.site.altitude.m),
            "region": obs.site.region,
        }

        if hasattr(obs, "atmosphere"):
            metadata["atmosphere"] = True
            metadata["pwv"] = float(np.round(obs.atmosphere.weather.pwv, 3))
            metadata["base_temperature"] = float(np.round(obs.atmosphere.weather.temperature[0], 3))
        else:
            metadata["atmosphere"] = False

        if hasattr(self, "map"):
            metadata["input_map"] = self.map

        return TOD(
            data=obs.loading,
            dets=obs.instrument.dets,
            coords=obs.coords,
            units="pW",
            metadata=metadata,
        )

    def plot_hits(self, x_bins=100, y_bins=100):
        self.plan.plot_hits(instrument=self.instrument, x_bins=x_bins, y_bins=y_bins)

    def __repr__(self):
        instrument_tree = "├ " + self.instrument.__repr__().replace("\n", "\n│ ")
        site_tree = "├ " + self.site.__repr__().replace("\n", "\n│ ")
        plan_tree = "├ " + self.plans.__repr__().replace("\n", "\n│ ")

        atmosphere_tree = f"├ atmosphere: {self.atmosphere}" if self.atmosphere else ""

        trees = []
        attrs = [attr for attr in ["cmb", "map"] if getattr(self, attr, None)]
        if attrs:
            for attr in attrs[:-1]:
                trees.append("├ " + getattr(self, attr).__repr__().replace("\n", "\n│ "))
            trees.append("└ " + getattr(self, attrs[-1]).__repr__().replace("\n", "\n  "))

        trees_string = "\n".join(trees)

        return f"""Simulation
{instrument_tree}
{site_tree}
{plan_tree}
{atmosphere_tree}
{trees_string}"""

    @property
    def total_loading(self):
        return sum([d for d in self.loading.values()])

    @property
    def min_time(self):
        return self.plans[0].start_time

    @property
    def max_time(self):
        return self.plans[-1].end_time
