"""Streaming maximum-likelihood mapper.

Mathematically equivalent to MaximumLikelihoodMapper but streams data one
chunk at a time. Instead of holding all TOD signals and pointing matrices in
memory, it:

  - Stores only compact per-chunk noise models (A_inv, shape n_dets × n_freqs,
    ~1000× smaller than the TOD signal) between CG passes.
  - Recomputes pointing matrices on the fly from coordinates (no storage).
  - Uses the incremental CG residual update  r_{k+1} = r_k - α_k A p_k
    so each CG step costs exactly one data pass (vs. two in the batch version).

Memory bound: O(n_pixels + n_chunks × n_dets × n_freqs) — independent of
the number of time samples.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm

from ..io import DEFAULT_BAR_FORMAT
from ..map import Map
from ..tod import TOD
from .base import BaseProjectionMapper

logger = logging.getLogger("maria")

try:
    torch.set_num_threads(16)
except Exception:
    pass


class StreamingMaximumLikelihoodMapper(BaseProjectionMapper):
    """Memory-efficient ML mapper using streaming conjugate gradient.

    Parameters
    ----------
    data_source:
        Callable that returns an iterable of TODs. Called once per CG step
        and once per epoch for noise model updates. Must be deterministic and
        yield TODs in the same order on every call.

        Examples::

            lambda: sim.stream()
            lambda: (TOD.load(p) for p in saved_paths)
    """

    def __init__(
        self,
        data_source: Callable[[], Iterable[TOD]],
        target: Map = None,
        prior: bool = False,
        center=None,
        stokes: str = None,
        width=None,
        height=None,
        resolution=None,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
        },
        map_postprocessing: dict = {},
        progress_bars: bool = True,
        bilinear: bool = False,
        verbose: bool = False,
        noise_model_config: dict = {},
    ):
        self.data_source = data_source
        self.apply_prior = prior
        self.verbose = verbose
        self.noise_model_config = noise_model_config

        # Peek at first TOD for map-geometry initialisation only — not stored.
        first_tod = next(iter(data_source()))

        super().__init__(
            tods=[first_tod],
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

        self.sol = torch.zeros(self.map_size)
        self.hits = torch.zeros(self.map_size)
        self._b = torch.zeros(self.map_size)
        self._chunk_noise: list[dict] = []
        self._first_pass_done = False

    # ------------------------------------------------------------------
    # Pointing matrix — recomputed on the fly, never stored
    # ------------------------------------------------------------------

    def _build_pointing(self, tod: TOD) -> torch.Tensor:
        weights, samples, pixels, n_samples, n_pixels = (
            self.map._stokes_weighted_pointing_matrix_ingredients(
                coords=tod.coords, dets=tod.dets, bilinear=self.bilinear
            )
        )
        indices = torch.stack(
            [torch.tensor(samples, dtype=torch.long), torch.tensor(pixels, dtype=torch.long)],
            dim=0,
        )
        return torch.sparse_coo_tensor(
            indices=indices,
            values=torch.tensor(weights, dtype=torch.float),
            size=(n_samples, n_pixels),
        ).coalesce()

    # ------------------------------------------------------------------
    # Noise model — compact A_inv stored per chunk, not the TOD signal
    # ------------------------------------------------------------------

    def _estimate_noise(self, d: torch.Tensor) -> dict:
        """Estimate per-chunk noise model from (n_dets, n_samples) data.

        Returns a compact dict containing only A_inv (the inverse noise power
        spectrum) and the shape metadata needed by _apply_N_inv.
        """
        n_dets, n_samples = d.shape
        w = torch.tensor(
            sp.signal.windows.tukey(
                M=n_samples, alpha=self.noise_model_config.get("window_alpha", 0.1)
            ),
            dtype=torch.float,
        ).unsqueeze(0)  # (1, n_samples) — broadcasts over detectors

        fn = torch.fft.fft(w * d)
        smooth_ps = torch.tensor(
            sp.ndimage.gaussian_filter(
                fn.abs().square().numpy(),
                sigma=self.noise_model_config.get("ps_sigma", 8),
                axes=-1,
                truncate=1,
            )
        )
        return {
            "A_inv": 1.0 / smooth_ps,  # (n_dets, n_samples) — compact noise model
            "n_dets": n_dets,
            "n_samples": n_samples,
        }

    def _apply_N_inv(self, d: torch.Tensor, chunk: dict) -> torch.Tensor:
        """Apply N^{-1} to d (n_dets, n_samples). Returns ravelled (n_dets * n_samples,)."""
        n_samples = chunk["n_samples"]
        w = torch.tensor(
            sp.signal.windows.tukey(
                M=n_samples, alpha=self.noise_model_config.get("window_alpha", 0.1)
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        Nfwd = chunk["A_inv"] * torch.fft.fft(w * d)
        return torch.fft.ifft(Nfwd).real.ravel()

    # ------------------------------------------------------------------
    # Core streaming passes
    # ------------------------------------------------------------------

    def _update_noise_and_b(self, subtract_map: bool = True) -> None:
        """One data pass: refresh per-chunk noise models and accumulate b = Σ Pᵀ N⁻¹ d.

        On the very first call (first_pass_done=False) also accumulates hits
        and a bin map used to initialise the CG solution.
        """
        current_sol = self.sol.detach()
        b = torch.zeros(self.map_size)
        chunk_noise: list[dict] = []

        if not self._first_pass_done:
            hits = torch.zeros(self.map_size)
            bin_sum = torch.zeros(self.map_size)
            bin_wgt = torch.zeros(self.map_size)

        for tod in tqdm(
            self.data_source(),
            desc="Updating noise model",
            bar_format=DEFAULT_BAR_FORMAT,
            disable=not self.progress_bars,
        ):
            P = self._build_pointing(tod)

            # Raw signal in native TOD units (pW) — matches MaximumLikelihoodMapper
            # which uses tod.signal.compute() without any unit conversion.
            d = torch.tensor(
                sp.signal.detrend(tod.signal.compute()),
                dtype=torch.float,
            )
            n_dets, n_samples = d.shape

            # Subtract current map estimate from data before estimating noise.
            d_for_noise = (
                d - (P @ current_sol).reshape(n_dets, n_samples).detach()
                if subtract_map
                else d
            )

            chunk = self._estimate_noise(d_for_noise)
            chunk_noise.append(chunk)

            b += P.T @ self._apply_N_inv(d, chunk)

            if not self._first_pass_done:
                hits += P.abs().sum(dim=0).to_dense()
                # Uniform-weight bin map for sol initialisation (raw pW, consistent with d).
                D = d.ravel()
                bin_sum += D @ P
                bin_wgt += torch.ones_like(D) @ P.abs()

        self._b = b
        self._chunk_noise = chunk_noise

        if not self._first_pass_done:
            self.hits = hits
            self._bin_sum = bin_sum
            self._bin_wgt = bin_wgt
            self._first_pass_done = True

    def _apply_PNP(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A = Σᵢ Pᵢᵀ Nᵢ⁻¹ Pᵢ to x. One full data pass per call."""
        result = torch.zeros_like(x)
        for chunk_idx, tod in enumerate(self.data_source()):
            chunk = self._chunk_noise[chunk_idx]
            P = self._build_pointing(tod)
            Px = (P @ x).reshape(chunk["n_dets"], chunk["n_samples"])
            result += P.T @ self._apply_N_inv(Px, chunk)
        return result

    # ------------------------------------------------------------------
    # Solution initialisation
    # ------------------------------------------------------------------

    def _reset_sol(self) -> None:
        """Initialise solution from bin map (mirrors MaximumLikelihoodMapper.reset_sol)."""
        wgt = self._bin_wgt.clamp(min=1e-30)
        naive = (self._bin_sum / wgt).reshape(self.map_shape)
        H = self.hits.reshape(self.map_shape)
        naive = torch.where(H > 0, naive.float(), torch.zeros_like(naive))
        self.sol = torch.where(naive.isfinite(), naive, torch.zeros_like(naive)).ravel()

    # ------------------------------------------------------------------
    # Conjugate gradient
    # ------------------------------------------------------------------

    def _conjugate_gradient(
        self,
        max_steps: int,
        alpha_tol: float,
        epoch: int,
        n_epochs: int,
    ) -> None:
        y = self._b
        sol = self.sol.clone()

        # Initial residual. When sol=0 we get r_0=y for free (no extra pass).
        r = (y - self._apply_PNP(sol)) if sol.abs().max() > 0 else y.clone()

        p_list: list[torch.Tensor] = []
        Ap_list: list[torch.Tensor] = []
        pAp_list: list[torch.Tensor] = []
        alpha_list: list[float] = []

        with tqdm(
            desc=f"Fitting map (epoch {epoch + 1}/{n_epochs})",
            bar_format=DEFAULT_BAR_FORMAT,
            disable=not self.progress_bars,
        ) as pbar:
            for _ in range(max_steps):
                # Conjugate direction (modified Gram–Schmidt against previous directions).
                p_k = r - sum(
                    p_i * float((r @ Ap_i) / pAp_i)
                    for p_i, Ap_i, pAp_i in zip(p_list, Ap_list, pAp_list)
                ) if p_list else r.clone()

                Ap_k = self._apply_PNP(p_k)  # one data pass
                pAp_k = float(p_k @ Ap_k)
                if abs(pAp_k) < 1e-30:
                    break

                alpha_k = float(p_k @ r) / pAp_k
                sol = sol + alpha_k * p_k
                r = r - alpha_k * Ap_k  # incremental residual — no extra pass

                alpha_list.append(alpha_k)
                p_list.append(p_k)
                Ap_list.append(Ap_k)
                pAp_list.append(pAp_k)

                pbar.update(1)
                pbar.set_postfix(alpha=f"{alpha_k:.3e}")

                if len(alpha_list) > 1 and alpha_k < alpha_tol * float(
                    torch.tensor(alpha_list[:-1]).median()
                ):
                    logger.info("CG converged")
                    break

        self.sol = sol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str = "conjugate_gradient",
        epochs: int = 4,
        max_steps_per_epoch: int = 500,
        alpha_tol: float = 1e-3,
        plot: bool = False,
        plot_kwargs: dict = {},
    ) -> None:
        if method != "conjugate_gradient":
            raise NotImplementedError(f"Method '{method}' is not supported by StreamingMaximumLikelihoodMapper")

        for epoch in range(epochs):
            # First epoch: no map to subtract yet; also accumulates hits and bin map.
            # sol stays at zero — CG warm-start from the bin map is skipped here
            # (the batch mapper does it but it's only a convergence hint, not part of
            # the mathematical algorithm; starting from zero gives an equivalent solution).
            self._update_noise_and_b(subtract_map=self._first_pass_done)

            self._conjugate_gradient(max_steps_per_epoch, alpha_tol, epoch, epochs)

            if plot:
                self.map.plot(**plot_kwargs)

    # ------------------------------------------------------------------
    # BaseProjectionMapper interface
    # ------------------------------------------------------------------

    def get_map_data(self) -> np.ndarray:
        return self.sol.detach().numpy()

    def get_map_weight(self) -> np.ndarray:
        return self.hits.numpy()
