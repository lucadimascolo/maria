from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ..units import Quantity
from .projection_map import ProjectionMap


def _extract_2d(m: ProjectionMap, stokes: str = "I", nu_index: int = 0, t_index: int = 0) -> np.ndarray:
    d = m.data
    if "stokes" in m.dims:
        d = d[list(m.stokes).index(stokes)]
    if "nu" in m.dims:
        d = d[nu_index]
    if "v" in m.dims:
        d = d[0]
    if "z" in m.dims:
        d = d[0]
    if "t" in m.dims:
        d = d[t_index]
    return np.asarray(d.compute(), dtype=float).reshape(m.dims["eta"], m.dims["xi"])


def compute_transfer_function_cross(
    input_map: ProjectionMap,
    output_map: ProjectionMap,
    n_bins: int = 30,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    window: str | bool | np.ndarray = "hann",
    taper: float = 0.1,
    pad_factor: int = 1,
    u_min: float | None = None,
    u_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally-averaged spatial transfer function via cross-correlation.

    Uses T(u) = Re⟨F_in*(u) · F_out(u)⟩ / ⟨|F_in(u)|²⟩ rather than the naive
    power ratio P_out/P_in. Because noise in the output map is uncorrelated with
    the input model, it averages to zero in the cross-spectrum numerator and does
    not bias the estimate.

    A value of 1 indicates perfect signal recovery at that angular scale; values
    below 1 indicate signal suppression (e.g. from filtering or atmospheric removal).

    Parameters
    ----------
    input_map : ProjectionMap
        The input sky map injected into the simulation.
    output_map : ProjectionMap
        The recovered map produced by a mapper.
    n_bins : int
        Number of logarithmically-spaced spatial frequency bins.
    stokes : str
        Stokes parameter to use ("I", "Q", "U", or "V").
    nu_index : int
        Frequency channel index for multi-channel maps.
    t_index : int
        Time index for time-varying maps.
    window : str, bool, or np.ndarray
        Apodization window applied before the FFT to reduce spectral leakage.
        - ``"tukey"``: separable Tukey window with
          cosine-tapered fraction ``taper`` on each edge; leaves the central
          region at unit weight.
        - ``"hann"`` (default) or ``True``: full Hann window (goes to zero at both edges).
        - ``np.ndarray`` of shape ``(ny, nx)``: custom 2D window applied as-is.
        - ``False`` or ``None``: no windowing.
    taper : float
        Fraction of each axis tapered by the cosine roll-off when
        ``window="tukey"``. Must be in [0, 1]. Default is 0.1.
    pad_factor : int
        Zero-pad each axis to ``pad_factor`` times its original length before
        the FFT. Increases the density of k-space samples (improves large-scale
        / low-wavenumber sampling) without changing the pixel size or Nyquist
        frequency. Must be >= 1. Default is 1 (no padding).

    Returns
    -------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values (dimensionless).
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")

    if output_map.units != input_map.units:
        output_map = output_map.to(input_map.units)

    f_in = _extract_2d(input_map.resample(output_map), stokes, nu_index, t_index)
    f_out = _extract_2d(output_map, stokes, nu_index, t_index)

    ny, nx = f_out.shape

    f_in = np.where(np.isfinite(f_in), f_in, 0.00)
    f_out = np.where(np.isfinite(f_out), f_out, 0.00)

    if window is not False and window is not None:
        if isinstance(window, np.ndarray):
            if window.shape != (ny, nx):
                raise ValueError(f"Custom window shape {window.shape} does not match map shape ({ny}, {nx})")
            win = window.astype(float)
        else:
            w_name = window if isinstance(window, str) else "hann"
            if w_name == "hann":
                wx, wy = np.hanning(nx), np.hanning(ny)
            elif w_name == "tukey":
                wx = sp.signal.windows.tukey(nx, alpha=taper)
                wy = sp.signal.windows.tukey(ny, alpha=taper)
            else:
                raise ValueError(f"Unsupported window type: {window}")
            win = np.outer(wy, wx)
        win /= np.nanmax(win)

        f_in = f_in * win
        f_out = f_out * win

    ny_pad, nx_pad = ny * pad_factor, nx * pad_factor
    F_in = np.fft.fftshift(np.fft.fft2(f_in, s=(ny_pad, nx_pad)))
    F_out = np.fft.fftshift(np.fft.fft2(f_out, s=(ny_pad, nx_pad)))

    # Cross-spectrum: noise in F_out is uncorrelated with F_in, so Re(F_in*·N)→0
    # when averaged, leaving only the signal contribution in the numerator.
    cross   = np.conj(F_in) * F_out
    P_den   = np.real(np.conj(F_in) * F_in)
    P_num   = np.real(cross)
    P_imag  = np.imag(cross)
    P_out   = np.real(np.conj(F_out) * F_out)

    dx = output_map.xi_res.rad
    dy = abs(output_map.eta_res.rad)
    kx = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny_pad, d=dy))
    K = np.hypot(*np.meshgrid(kx, ky))

    k_min = u_min if u_min is not None else max(1.0 / (nx_pad * dx), 1.0 / (ny_pad * dy))
    k_max = u_max if u_max is not None else 0.5 * min(1.0 / dx, 1.0 / dy)
    bins = np.geomspace(k_min, k_max, n_bins + 1)
    u = np.sqrt(bins[:-1] * bins[1:])

    bin_idx    = np.digitize(K.ravel(), bins) - 1
    mask       = (bin_idx >= 0) & (bin_idx < n_bins)
    sum_P_den  = np.bincount(bin_idx[mask], weights=P_den.ravel()[mask],  minlength=n_bins)
    sum_P_num  = np.bincount(bin_idx[mask], weights=P_num.ravel()[mask],  minlength=n_bins)
    sum_P_imag = np.bincount(bin_idx[mask], weights=P_imag.ravel()[mask], minlength=n_bins)
    sum_P_out  = np.bincount(bin_idx[mask], weights=P_out.ravel()[mask],  minlength=n_bins)

    T = np.where(sum_P_den > 0, sum_P_num / sum_P_den, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(
            (sum_P_den > 0) & (sum_P_out > 0),
            (sum_P_num**2 + sum_P_imag**2) / (sum_P_den * sum_P_out),
            np.nan,
        )

    return u, T, coh


def compute_transfer_function_auto(
    input_map: ProjectionMap,
    output_map: ProjectionMap,
    n_bins: int = 30,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    window: str | bool | np.ndarray = "hann",
    taper: float = 0.1,
    pad_factor: int = 1,
    u_min: float | None = None,
    u_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naive P_out/P_in transfer-function estimator.

    T(u) = ⟨|F_out(u)|²⟩ / ⟨|F_in(u)|²⟩

    Noise in F_out biases this estimator high; use only for noiseless checks
    or as a quick sanity estimate. Coherence is not defined here → NaN.
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")

    if output_map.units != input_map.units:
        output_map = output_map.to(input_map.units)

    f_in  = _extract_2d(input_map.resample(output_map), stokes, nu_index, t_index)
    f_out = _extract_2d(output_map, stokes, nu_index, t_index)

    ny, nx = f_out.shape

    f_in  = np.where(np.isfinite(f_in),  f_in,  0.0)
    f_out = np.where(np.isfinite(f_out), f_out, 0.0)

    if window is not False and window is not None:
        if isinstance(window, np.ndarray):
            if window.shape != (ny, nx):
                raise ValueError(f"Custom window shape {window.shape} does not match map shape ({ny}, {nx})")
            win = window.astype(float)
        else:
            w_name = window if isinstance(window, str) else "hann"
            if w_name == "hann":
                wx, wy = np.hanning(nx), np.hanning(ny)
            elif w_name == "tukey":
                wx = sp.signal.windows.tukey(nx, alpha=taper)
                wy = sp.signal.windows.tukey(ny, alpha=taper)
            else:
                raise ValueError(f"Unsupported window type: {window}")
            win = np.outer(wy, wx)
        win /= np.nanmax(win)
        f_in  = f_in  * win
        f_out = f_out * win

    ny_pad, nx_pad = ny * pad_factor, nx * pad_factor
    F_in  = np.fft.fftshift(np.fft.fft2(f_in,  s=(ny_pad, nx_pad)))
    F_out = np.fft.fftshift(np.fft.fft2(f_out, s=(ny_pad, nx_pad)))

    P_in  = np.real(np.conj(F_in)  * F_in)
    P_out = np.real(np.conj(F_out) * F_out)

    dx = output_map.xi_res.rad
    dy = abs(output_map.eta_res.rad)
    kx = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny_pad, d=dy))
    K  = np.hypot(*np.meshgrid(kx, ky))

    k_min = u_min if u_min is not None else max(1.0 / (nx_pad * dx), 1.0 / (ny_pad * dy))
    k_max = u_max if u_max is not None else 0.5 * min(1.0 / dx, 1.0 / dy)
    bins  = np.geomspace(k_min, k_max, n_bins + 1)
    u     = np.sqrt(bins[:-1] * bins[1:])

    bin_idx   = np.digitize(K.ravel(), bins) - 1
    mask      = (bin_idx >= 0) & (bin_idx < n_bins)
    sum_P_in  = np.bincount(bin_idx[mask], weights=P_in.ravel()[mask],  minlength=n_bins)
    sum_P_out = np.bincount(bin_idx[mask], weights=P_out.ravel()[mask], minlength=n_bins)

    T = np.where(sum_P_in > 0, sum_P_out / sum_P_in, np.nan)
    return u, T, np.full(n_bins, np.nan)


def _mcm(mask: np.ndarray, K: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Flat-sky mode-coupling matrix from an apodised mask.

    M[i,j] = (1/N_i) Σ_{k∈i} Σ_{k'∈j} |Ŵ(k−k')|²  where Ŵ = FFT(mask)/N_pix.
    Computed via FFT convolution: h_j = IFFT(FFT(P_W)·FFT(f_j)), then M[i,j] = dot(f_i, h_j)/N_i.
    Full-sky mask (all ones) → M = identity.
    """
    ny, nx  = mask.shape
    n_pix   = ny * nx
    n_bins  = len(bins) - 1
    P_W     = np.abs(np.fft.fft2(mask) / n_pix) ** 2
    F_PW    = np.fft.fft2(P_W)
    k_flat  = K.ravel()
    bin_idx = np.digitize(k_flat, bins) - 1
    N       = np.array([(bin_idx == b).sum() for b in range(n_bins)])
    M       = np.zeros((n_bins, n_bins))
    for j in range(n_bins):
        if N[j] == 0:
            continue
        f_j = (bin_idx == j).reshape(ny, nx).astype(float)
        h_j = np.real(np.fft.ifft2(F_PW * np.fft.fft2(f_j)))
        for i in range(n_bins):
            if N[i] == 0:
                continue
            f_i    = (bin_idx == i).reshape(ny, nx).astype(float)
            M[i,j] = np.dot(f_i.ravel(), h_j.ravel()) / N[i]
    return M


def compute_transfer_function_pcl(
    input_map: ProjectionMap,
    output_map: ProjectionMap,
    n_bins: int = 20,
    stokes: str = "I",
    nu_index: int = 0,
    t_index: int = 0,
    u_min: float | None = None,
    u_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-Cl / MASTER transfer-function estimator.

    Computes the mode-coupling matrix M from the binary coverage mask and solves
    M · C_true = C_pseudo, giving T(u) = [M⁻¹C^cross]_u / [M⁻¹C^in]_u.
    Valid bins (N_modes > 0) are solved via least-squares; empty bins → NaN.

    Parameters
    ----------
    u_min, u_max : float or None
        Spatial-frequency range in cycles/rad (None = map-geometry limits).
    """
    if output_map.units != input_map.units:
        output_map = output_map.to(input_map.units)

    f_in  = _extract_2d(input_map.resample(output_map), stokes, nu_index, t_index)
    f_out = _extract_2d(output_map, stokes, nu_index, t_index)

    f_in  = np.where(np.isfinite(f_in),  f_in,  0.0)
    f_out = np.where(np.isfinite(f_out), f_out, 0.0)

    # Binary coverage mask
    ny, nx = f_out.shape
    w    = getattr(output_map, "_weight", None)
    mask = (np.asarray(w).squeeze() > 0).astype(float) if w is not None else np.ones((ny, nx))

    f_in  = f_in  * mask
    f_out = f_out * mask

    dx = output_map.xi_res.rad
    dy = abs(output_map.eta_res.rad)
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    K  = np.hypot(*np.meshgrid(kx, ky))

    k_min = u_min if u_min is not None else max(1.0 / (nx * dx), 1.0 / (ny * dy))
    k_max = u_max if u_max is not None else 0.5 * min(1.0 / dx, 1.0 / dy)
    bins  = np.geomspace(k_min, k_max, n_bins + 1)
    u     = np.sqrt(bins[:-1] * bins[1:])

    bin_idx = np.digitize(K.ravel(), bins) - 1
    N_modes = np.array([(bin_idx == b).sum() for b in range(n_bins)])

    F_in  = np.fft.fft2(f_in)
    F_out = np.fft.fft2(f_out)

    def _bin_avg(P):
        flat = P.ravel()
        return np.array([flat[bin_idx == b].mean() if N_modes[b] > 0 else 0.0
                         for b in range(n_bins)])

    cross         = np.conj(F_in) * F_out
    C_in          = _bin_avg(np.real(np.conj(F_in) * F_in))
    C_cross       = _bin_avg(np.real(cross))
    C_cross_imag  = _bin_avg(np.imag(cross))
    C_out         = _bin_avg(np.real(np.conj(F_out) * F_out))

    M     = _mcm(mask, K, bins)
    valid = N_modes > 0
    T     = np.full(n_bins, np.nan)
    coh   = np.full(n_bins, np.nan)

    if valid.sum() >= 2:
        M_s              = M[np.ix_(valid, valid)]
        Ci,  *_          = np.linalg.lstsq(M_s, C_in[valid],         rcond=None)
        Cc,  *_          = np.linalg.lstsq(M_s, C_cross[valid],      rcond=None)
        Cci, *_          = np.linalg.lstsq(M_s, C_cross_imag[valid], rcond=None)
        Co,  *_          = np.linalg.lstsq(M_s, C_out[valid],        rcond=None)
        with np.errstate(invalid="ignore", divide="ignore"):
            T[valid]   = np.where(np.abs(Ci) > 0, Cc / Ci, np.nan)
            coh[valid] = np.where(
                (np.abs(Ci) > 0) & (Co > 0),
                (Cc**2 + Cci**2) / (Ci * Co),
                np.nan,
            )

    return u, T, coh


class TransferFunction:
    """Result of a spatial transfer function computation.

    Attributes
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian, shape ``(n_bins,)``.
    T : np.ndarray
        Transfer function values, shape ``(n_nu, n_bins)``.
    nu : Quantity or None
        Frequency axis corresponding to the first dimension of ``T``.
    beam_fwhm : np.ndarray or None
        Per-channel beam FWHM in radians, shape ``(n_nu,)``.
    """

    def __init__(self, u, T, nu=None, beam_fwhm=None, coherence=None, input_map=None, output_map=None):
        self.u = u
        self.T = T
        self.nu = nu
        self.beam_fwhm = beam_fwhm
        self.coherence = coherence
        self.input_map = input_map
        self.output_map = output_map

    def plot(self, ax=None, x_unit="arcmin", filename=None, add_beam=True, slices=None):
        """Plot the transfer function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        x_unit : str
            Angular unit for the x-axis: ``"arcsec"``, ``"arcmin"``, or ``"deg"``.
        filename : str, optional
            Save the figure to this path when provided.
        add_beam : bool
            Overlay the theoretical Gaussian beam curve per channel.
        slices : dict, optional
            Channel selection, e.g. ``dict(nu=[0, 2])``.  ``None`` plots all
            channels, consistent with how ``slices`` is used in map ``.plot()``.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        idx = list(np.atleast_1d(slices["nu"])) if (slices and "nu" in slices) else slice(None)
        return plot_transfer_function(
            self.u,
            self.T[idx],
            nu=self.nu[idx] if self.nu is not None else None,
            beam_fwhm=self.beam_fwhm[idx] if (add_beam and self.beam_fwhm is not None) else None,
            ax=ax,
            x_unit=x_unit,
            filename=filename,
        )

    def __repr__(self):
        n_nu = self.T.shape[0]
        nu_str = f"  nu: {self.nu}\n" if self.nu is not None else ""
        return (
            f"TransferFunction:\n"
            f"  channels: {n_nu}\n"
            f"{nu_str}"
            f"  bins: {len(self.u)}\n"
            f"  u: [{self.u.min():.3g}, {self.u.max():.3g}] cycles/rad\n"
            f"  T: [{np.nanmin(self.T):.3f}, {np.nanmax(self.T):.3f}]"
        )


_RAD_TO_DEG = 180.0 / np.pi
_ANGULAR_UNITS = {
    "arcsec": 3600.00 * _RAD_TO_DEG,
    "arcmin": 60.00 * _RAD_TO_DEG,
    "deg": _RAD_TO_DEG,
    "rad": 1.0,
}


def plot_transfer_function(
    u: np.ndarray,
    T: np.ndarray,
    nu=None,
    beam_fwhm=None,
    ax=None,
    x_unit: str = "arcmin",
    filename: str = None,
) -> plt.Axes:
    """
    Plot the spatial transfer function.

    Parameters
    ----------
    u : np.ndarray
        Spatial frequency bin centres in cycles per radian.
    T : np.ndarray
        Transfer function values, shape ``(n_nu, n_bins)``.
    nu : Quantity, optional
        Frequency axis for labelling each curve.
    beam_fwhm : array-like of float, optional
        Per-channel beam FWHM in radians, shape ``(n_nu,)``.  When provided, a
        dashed Gaussian beam curve is overlaid for each channel.
    ax : matplotlib.axes.Axes, optional
    x_unit : str
        Angular unit for the x-axis: ``"arcsec"``, ``"arcmin"`` (default), or ``"deg"``.
    filename : str, optional
        Save the figure to this path when provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    factor = _ANGULAR_UNITS.get(x_unit, _ANGULAR_UNITS["arcmin"])
    theta = factor / u
    n_nu = T.shape[0]
    colors = mpl.colormaps["viridis"](np.linspace(0.2, 0.85, n_nu)) if n_nu > 1 else ["steelblue"]
    u_dense = np.geomspace(u.min(), u.max(), 500) if beam_fwhm is not None else None

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", zorder=0)

    for i, (T_row, color) in enumerate(zip(T, colors)):
        label = str(Quantity(nu[i], "Hz")) if nu is not None else "Measured"
        ax.plot(theta, T_row, color=color, lw=1.5, marker="o", ms=3, label=label)

        if beam_fwhm is not None:
            fwhm = float(beam_fwhm[i])
            if fwhm > 0:
                B = np.exp(-(np.pi**2) * fwhm**2 * u_dense**2 / (4.0 * np.log(2.0)))
                ax.plot(factor / u_dense, B, color=color, lw=1.5, ls="--")

    ax.legend(frameon=False, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"Angular scale [{x_unit}]")
    ax.set_ylabel("Transfer function")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(theta.min(), theta.max())

    if filename is not None:
        ax.get_figure().savefig(filename, dpi=150)

    return ax
