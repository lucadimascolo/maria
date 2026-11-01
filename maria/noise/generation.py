from __future__ import annotations

import numpy as np

from ..utils import compute_diameter, generate_spatial_basis

DEFAULT_NOISE_SIM_KWARGS = {"corr_prop": 0.5, "spatial_scale": 0.9}


def generate_noise_with_knee(
    shape: tuple,
    sample_rate: float = 1.0,
    knee: float = 0.0,
    beta: float = 1.0,
    basis: float = None,
    corr_prop: float = 0.0,
    seed: int = None,
):
    """
    Simulate noise with a unit NEP (a per-sample standard deviation of sqrt(sample_rate)) and a knee.

    A fraction 'corr_prop' of the pink noise power comes from pink-only modes, mixed in by the rows of 'basis'
    (n_dets, n_modes) normalized to 1. The noise is drawn from NumPy's global random state, or from its own
    generator if a seed is given.
    """

    rng = np.random if seed is None else np.random.default_rng(seed)

    noise = np.sqrt(sample_rate) * rng.standard_normal(shape)

    # pink noise
    if knee > 0:
        f = np.fft.fftfreq(n=shape[-1], d=1 / sample_rate)
        # a two-sided PSD of (knee / |f|) ** beta (in units of the white level), which equals the white noise at the
        # knee for any beta
        with np.errstate(divide="ignore", invalid="ignore"):
            pink_noise_power_spectrum = np.where(f != 0, 0.5 * (knee / np.abs(f)) ** beta, 0)

        weights = np.sqrt(2 * sample_rate * pink_noise_power_spectrum)

        def pink(shape):
            return np.real(np.fft.ifft(weights * np.fft.fft(rng.standard_normal(shape))))

        pink_noise = pink(shape)

        if basis is not None:
            # pink-only modes, so that the white noise of each detector is independent, with unit-norm rows, so
            # that every detector keeps the pink noise level set by the knee
            basis = basis / np.linalg.norm(basis, axis=-1, keepdims=True)
            noise_modes = pink((basis.shape[-1], shape[-1]))

            pink_noise = np.sqrt(corr_prop) * basis @ noise_modes + np.sqrt(1 - corr_prop) * pink_noise

        noise += pink_noise

    return noise


def generate_2d_fourier_noise(nx: float = 1024, ny: float = 1024, k0: float = 5e0, beta: float = 8 / 3):
    kx = np.fft.fftfreq(nx, d=1 / nx)
    ky = np.fft.fftfreq(ny, d=1 / ny)
    KY, KX = np.meshgrid(ky, kx)
    P = np.sqrt(k0**2 + KX**2 + KY**2) ** (-beta - 1)
    F = np.fft.fft2(np.sqrt(P) * np.fft.ifft2(np.random.standard_normal(size=(ny, nx)))).real

    return (F - F.mean()) / F.std()
