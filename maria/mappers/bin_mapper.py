from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..coords import infer_center_width_height
from ..io import DEFAULT_BAR_FORMAT, humanize_time, repr_phi_theta
from ..map import Map, ProjectionMap
from ..tod import TOD
from ..units import Quantity
from .base import BaseProjectionMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BinMapper(BaseProjectionMapper):
    def __init__(
        self,
        tods: Sequence[TOD],
        target: Map = None,
        center: tuple[float, float] = None,
        stokes: str = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
        progress_bars: bool = True,
        bilinear: bool = False,
        mpi: bool = False,
    ):
        # Set up MPI before super().__init__ so the comm is available for the
        # nu-grid synchronization that must happen before map arrays are sized.
        self._mpi = mpi
        self._mpi_comm = None
        self._mpi_SUM = None
        if mpi:
            try:
                from mpi4py import MPI as _MPI

                self._mpi_comm = _MPI.COMM_WORLD
                self._mpi_SUM = _MPI.SUM
            except ImportError as e:
                raise ImportError("mpi4py is required for MPI support: pip install mpi4py") from e

        super().__init__(
            tods=tods,
            target=target,
            stokes=stokes,
            center=center,
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=degrees,
            progress_bars=progress_bars,
            bilinear=bilinear,
        )

        # After add_tods() populates self.nu, synchronize
        # the frequency grid across all ranks.
        if mpi and self._mpi_comm is not None:
            all_nu_hz = self._mpi_comm.allgather(list(self.nu.Hz))
            combined_hz = sorted({v for row in all_nu_hz for v in row})
            if list(self.nu.Hz) != combined_hz:
                self.nu = Quantity(combined_hz, "Hz")
                beam_sum = np.zeros((len(self.nu), 1, 3))
                beam_wgt = np.zeros((len(self.nu), 1, 3))
                for nu_index, nu in enumerate(self.nu):
                    for tod in self.tods:
                        mask = tod.dets.band_center == nu.Hz
                        if mask.sum() > 0:
                            beam_sum[nu_index] += tod.duration * tod.dets.beams[mask].mean(axis=0)
                            beam_wgt[nu_index] += tod.duration
                with np.errstate(invalid="ignore"):
                    self.beam = np.where(beam_wgt > 0, beam_sum / beam_wgt, 0.0)

        self.products = {
            "data": np.zeros(self.map_shape),
            "weight": np.ones(self.map_shape),
        }

        self.has_been_ran = False

    @property
    def map(self):
        if not self.has_been_ran:
            raise RuntimeError("Mapper has not been run yet!")
        return super().map

    def get_map_data(self):
        return self.products["data"]

    def get_map_weight(self):
        return self.products["weight"]

    def run(self):
        """
        Run the BinMapper
        """

        map_sum = np.zeros(self.map_size)
        map_wgt = np.zeros(self.map_size)

        pbar = tqdm(
            self.tods,
            total=len(self.tods),
            desc=f"Mapping",
            postfix={"tod": f"1/{len(self.tods)}"},
            bar_format=DEFAULT_BAR_FORMAT,
            disable=not self.progress_bars,
        )

        for tod in pbar:
            if not tod.shape[0] > 0:
                continue

            P = super().map.stokes_weighted_pointing_matrix(coords=tod.coords, dets=tod.dets, bilinear=self.bilinear)
            D = tod.signal.compute().ravel()
            W = tod.weight.compute().ravel()

            map_sum += (W * D) @ P
            map_wgt += W @ np.abs(P)

        if self._mpi:
            total_sum = np.zeros_like(map_sum)
            total_wgt = np.zeros_like(map_wgt)
            self._mpi_comm.Allreduce(map_sum, total_sum, op=self._mpi_SUM)
            self._mpi_comm.Allreduce(map_wgt, total_wgt, op=self._mpi_SUM)
            map_sum = total_sum
            map_wgt = total_wgt

        self.products = {
            "data": map_sum / map_wgt,
            "weight": map_wgt,
            "sum": map_sum,
        }

        self.has_been_ran = True

        return self.map
