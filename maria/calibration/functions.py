import numpy as np
import scipy as sp

from ..constants import T_CMB, h, k_B
from ..errors import ShapeError
from ..functions.radiometry import (
    inverse_planck_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    rayleigh_jeans_spectrum,
)  # noqa


def identity(x: float, **kwargs):
    return x


def cmb_temperature_anisotropy_to_compton_y(dT_CMB, nu, **kwargs):
    x = h * nu / (k_B * T_CMB)
    f = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4
    return dT_CMB / (f * T_CMB)


def compton_y_to_cmb_temperature_anisotropy(y, nu, **kwargs):
    x = h * nu / (k_B * T_CMB)
    f = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4
    return y * f * T_CMB


def cmb_temperature_anisotropy_to_brightness_temperature(dT_CMB, **kwargs):
    return dT_CMB + T_CMB


def brightness_temperature_to_cmb_temperature_anisotropy(T_b, **kwargs):
    return T_b - T_CMB


def rayleigh_jeans_temperature_to_spectral_radiance(T_RJ, nu, **kwargs):
    return rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu)


def spectral_radiance_to_rayleigh_jeans_temperature(I_nu, nu, **kwargs):
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_brightness_temperature(T_RJ, nu, **kwargs):
    I_nu = rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu)
    return inverse_planck_spectrum(I_nu=I_nu, nu=nu)


def _check_brightness_temperature(T_b):
    """
    A brightness temperature is absolute: the Planck spectrum of a negative one is negative and does not tend to 0
    (about -7.2 K_RJ at 150 GHz for -1e-6 K_b), so negative values raise.
    """
    if np.any(np.asarray(T_b) < 0):
        raise ValueError("Brightness temperatures (K_b) are absolute and must not be negative.")


def brightness_temperature_to_rayleigh_jeans_temperature(T_b, nu, **kwargs):
    _check_brightness_temperature(T_b)
    I_nu = planck_spectrum(T_b=T_b, nu=nu)
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_power(
    T_RJ,
    band,
    polarized: bool = False,
    spectrum=None,
    **kwargs,
):
    """
    nu: frequency, in Hz
    band: response to a Rayleigh-Jeans source
    """
    if spectrum:
        integral = band.compute_transmission_integral(
            spectrum=spectrum,
            pwv=kwargs["pwv"],
            effective_temperature=kwargs["effective_temperature"],
            elevation=kwargs["elevation"],
        )

    else:
        integral = band.compute_transmission_integral(spectrum=None)

    return np.where(polarized, 0.5, 1.0) * k_B * integral * T_RJ


def power_to_rayleigh_jeans_temperature(P, band, polarized: bool = False, spectrum=None, **kwargs):
    """
    nu: frequency, in Hz
    passband: response to a Rayleigh-Jeans source
    """

    if spectrum:
        integral = band.compute_transmission_integral(
            spectrum=spectrum,
            pwv=kwargs["pwv"],
            effective_temperature=kwargs["effective_temperature"],
            elevation=kwargs["elevation"],
        )

    else:
        integral = band.compute_transmission_integral(spectrum=None)

    return P / (np.where(polarized, 0.5, 1.0) * k_B * integral)


# the number of brightness temperatures on which brightness_temperature_to_power computes the Planck response
_T_B_GRID_SIZE = 256


def _transmission_integral_samples(T_b, band, spectrum):
    """
    The integral of the RJ temperature through the band and the atmosphere, on the (T_eff, pwv, airmass) grid of
    the spectrum, for each T_b: it has shape (*spectrum.shape, T_b.size).
    """
    # add dimensions to broadcast with the atmospheric spectrum
    shaped_T_b = np.expand_dims(T_b, axis=(0, 1, 2, 3))
    shaped_nu = np.expand_dims(spectrum.side_nu.Hz, -1)
    sample_T_RJ = inverse_rayleigh_jeans_spectrum(
        planck_spectrum(T_b=shaped_T_b, nu=shaped_nu),
        nu=shaped_nu,
    )

    return np.trapezoid(
        y=sample_T_RJ * (np.exp(-np.expand_dims(spectrum.fields["opacity"], -1)) * band.passband(shaped_nu)),
        x=spectrum.side_nu.Hz,
        axis=-2,
    )


def brightness_temperature_to_power_explicit(T_b, band, polarized: bool = False, spectrum=None, eps: float = 1e-4, **kwargs):
    if T_b.ndim > 1:
        raise ShapeError("'T_b' must be one-dimensional")

    # stokes_factor = 0.5 if polarized else 1.0
    shaped_T_b = np.expand_dims(T_b, axis=0)

    if spectrum:
        # this has shape (spectrum.shape, T_b.size)
        integral_samples = _transmission_integral_samples(T_b, band, spectrum)

        points = spectrum.points[:3]
        xi = (
            kwargs["effective_temperature"],
            kwargs["pwv"],
            1 / np.sin(kwargs["elevation"]),
        )
        integral = sp.interpolate.RegularGridInterpolator(points, integral_samples)(xi)

    else:
        shaped_nu = np.expand_dims(band.nu.Hz, -1)
        sample_T_RJ = inverse_rayleigh_jeans_spectrum(
            planck_spectrum(T_b=shaped_T_b, nu=shaped_nu),
            nu=shaped_nu,
        )
        integral = np.trapezoid(y=sample_T_RJ * band.passband(shaped_nu), x=band.nu.Hz, axis=-2)

    # 'polarized' can be per detector, with the shape of the other kwargs (the last axis of 'integral' is T_b)
    return np.expand_dims(np.where(polarized, 0.5, 1.0), -1) * k_B * integral


def brightness_temperature_to_power(T_b, band, polarized: bool = False, spectrum=None, eps: float = 1e-4, **kwargs):
    """
    The power from a source with brightness temperature T_b. The (non-linear) Planck response is computed
    exactly on a grid of temperatures spanning [min(T_b), max(T_b)] and interpolated (it used to be the tangent
    line at min(T_b)). Non-finite temperatures give NaN.
    """
    T_b = np.asarray(T_b, dtype=float)
    _check_brightness_temperature(T_b)
    finite = np.isfinite(T_b)
    # the grid spans the finite temperatures (any grid will do when there are none: they all give NaN)
    T_b_finite = T_b[finite] if finite.any() else np.zeros(1)
    T_b_min, T_b_max = np.min(T_b_finite), np.max(T_b_finite)

    if T_b_max - T_b_min < eps:
        # clipped at 0 K: the Planck power of a negative temperature is negative (about -0.8 pW at 150 GHz),
        # and always reaching T_b_max, so that no value is outside the grid
        T_b_lo = max(T_b_min - eps / 2, 0.0)
        T_b_grid = np.array([T_b_lo, max(T_b_lo + eps, T_b_max)])
        T_b_floor = np.inf
    else:
        # geometric from the lowest positive T_b, or from a floor far below h nu / k_B (7.2 K at 150 GHz) where
        # the power is negligible, so that a tiny or zero T_b_min does not stretch the grid over many decades; a
        # T_b_min below the floor is added (a linear grid from 0 K to 1000 K gave 1.7x the power at T_CMB, and
        # a geometric one from 1e-14 K 1.6x at 850 GHz)
        T_b_floor = min(max(np.min(T_b_finite[T_b_finite > 0]), 1e-3), T_b_max / 2)
        T_b_grid = np.geomspace(T_b_floor, T_b_max, _T_B_GRID_SIZE - (T_b_min < T_b_floor))
        if T_b_min < T_b_floor:
            T_b_grid = np.append(T_b_min, T_b_grid)

    # the grid cell of each temperature
    T = np.clip(np.where(finite, T_b, T_b_min), T_b_grid[0], T_b_grid[-1])
    i = np.clip(np.searchsorted(T_b_grid, T, side="right") - 1, 0, T_b_grid.size - 2)
    T_0, T_1 = T_b_grid[i], T_b_grid[i + 1]
    # above the floor, log(P / T_b) is interpolated linearly in 1 / T_b, which is exact to first order in both the
    # Rayleigh-Jeans (P ~ T_b) and the Wien (log P ~ -h nu / k_B T_b) limits: linear in P, the geometric grid from
    # 1e-3 K gave 11% too much power at 1 K at 850 GHz. Below the floor the power is negligible, and linear in T_b.
    geometric = T_0 >= T_b_floor
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(geometric, (1 / T_0 - 1 / T) / (1 / T_0 - 1 / T_1), (T - T_0) / (T_1 - T_0))

    stokes_factor = np.where(polarized, 0.5, 1.0)

    if spectrum:
        # interpolate over (T_eff, pwv, airmass) on the spectrum grid at the two grid temperatures of each T_b,
        # so that the output is no larger than T_b
        integral_samples = k_B * _transmission_integral_samples(T_b_grid, band, spectrum)
        interpolator = sp.interpolate.RegularGridInterpolator(
            [*spectrum.points[:3], np.arange(T_b_grid.size)], integral_samples
        )
        xi = (kwargs["effective_temperature"], kwargs["pwv"], 1 / np.sin(kwargs["elevation"]), i)
        shape = np.broadcast_shapes(*[np.shape(x) for x in xi])
        xi = np.stack([np.broadcast_to(x, shape) for x in xi], axis=-1)
        P_0 = interpolator(xi).reshape(shape)
        xi[..., -1] += 1
        P_1 = interpolator(xi).reshape(shape)
    else:
        P_grid = brightness_temperature_to_power_explicit(T_b=T_b_grid, band=band, polarized=False, spectrum=None)
        P_0, P_1 = P_grid[i], P_grid[i + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(geometric, T * (P_0 / T_0) ** (1 - w) * (P_1 / T_1) ** w, (1 - w) * P_0 + w * P_1)
    P = np.where(finite, P, np.nan)

    return stokes_factor * P


def power_to_brightness_temperature(P, **kwargs):
    raise NotImplementedError()


def rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel(T_RJ: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Kelvin
    nu: frequency, in Hz
    pixel_area: Pixel area, in steradians
    """
    return rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu) * pixel_area


def spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature(E: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Jy/pixel
    nu: frequency, in Hz
    pixel_area: Pixel area, in steradians
    """
    I_nu = E / pixel_area
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_spectral_flux_density_per_beam(T_RJ: float, nu: float, beam_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Kelvin
    nu: frequency, in Hz
    beam_area: Beam area, in steradians
    """
    return rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu) * beam_area


def spectral_flux_density_per_beam_to_rayleigh_jeans_temperature(E: float, nu: float, beam_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Jy/pixel
    nu: frequency, in Hz
    beam_area: Beam area, in steradians
    """
    I_nu = E / beam_area
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def dP_dT_CMB(band, polarized=False, spectrum=None, eps=1e-4, **kwargs):
    sample_T_b = np.array([T_CMB - eps / 2, T_CMB + eps / 2])
    P = brightness_temperature_to_power_explicit(T_b=sample_T_b, band=band, polarized=polarized, spectrum=spectrum, **kwargs)
    return (P[..., 1] - P[..., 0]) / eps


def cmb_temperature_anisotropy_to_power(T_b, band, polarized=False, spectrum=None, **kwargs):
    return T_b * dP_dT_CMB(band=band, polarized=polarized, spectrum=spectrum, **kwargs)


def power_to_cmb_temperature_anisotropy(P, band, polarized=False, spectrum=None, **kwargs):
    return P / dP_dT_CMB(band=band, polarized=polarized, spectrum=spectrum, **kwargs)


def T_RJ_per_T_CMB(
    band,
    eps: float = 1e-3,
    **kwargs,
):
    """
    Color correction for NO ATMOSPHERE
    """
    test_T_b = T_CMB + np.array([[-eps / 2], [+eps / 2]])
    T_RJ = inverse_rayleigh_jeans_spectrum(
        planck_spectrum(T_b=test_T_b, nu=band.nu.Hz),
        nu=band.nu.Hz,
    )
    P = k_B * np.trapezoid(T_RJ * band.passband(band.nu.Hz), x=band.nu.Hz, axis=-1)
    return power_to_rayleigh_jeans_temperature((P[1] - P[0]) / eps, spectrum=None, band=band)


def cmb_temperature_anisotropy_to_rayleigh_jeans_temperature(
    dT_CMB: float,
    nu,
    eps: float = 1e-3,
    **kwargs,
):
    dI_dT_CMB = (planck_spectrum(T_CMB + eps / 2, nu=nu) - planck_spectrum(T_CMB - eps / 2, nu=nu)) / eps
    return inverse_rayleigh_jeans_spectrum(dT_CMB * dI_dT_CMB, nu=nu)


def rayleigh_jeans_temperature_to_cmb_temperature_anisotropy(
    T_RJ: float,
    nu,
    eps: float = 1e-3,
    **kwargs,
):
    dI_dT_CMB = (planck_spectrum(T_CMB + eps / 2, nu=nu) - planck_spectrum(T_CMB - eps / 2, nu=nu)) / eps
    return rayleigh_jeans_spectrum(T_RJ, nu=nu) / dI_dT_CMB


def spectral_radiance_to_spectral_flux_density_per_pixel(E: float, pixel_area: float, **kwargs):
    """
    E: Spectral flux density per area, in Jy/sr
    pixel_area: pixel area, in steradians
    """
    return E * pixel_area


def spectral_flux_density_per_pixel_to_spectral_radiance(E: float, pixel_area: float, **kwargs):
    """
    E: Spectral flux density per pixel, in Jy/pixel
    pixel_area: pixel area, in steradians
    """
    return E / pixel_area


def spectral_flux_density_per_beam_to_spectral_flux_density_per_pixel(
    E: float, pixel_area: float, beam_area: float, **kwargs
):
    """
    E: Spectral flux density per area, in Jy/sr
    pixel_area: pixel area, in steradians
    """
    return E * pixel_area / beam_area


def spectral_flux_density_per_pixel_to_spectral_flux_density_per_beam(
    E: float, pixel_area: float, beam_area: float, **kwargs
):
    """
    E: Spectral flux density per pixel, in Jy/pixel
    pixel_area: pixel area, in steradians
    """
    return E * beam_area / pixel_area
