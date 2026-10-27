from __future__ import annotations

import tracemalloc
from types import SimpleNamespace

import maria
import numpy as np
import pytest
from maria.calibration import Calibration
from maria.calibration.calibration import parse_calibration_signature
from maria.calibration.functions import brightness_temperature_to_power, brightness_temperature_to_power_explicit
from maria.constants import T_CMB, h, k_B
from maria.functions.radiometry import inverse_rayleigh_jeans_spectrum, planck_spectrum
from maria.spectrum import AtmosphericSpectrum


@pytest.mark.parametrize("signature", ["K_RJ to pW", "K_RJ -> pW -> K_CMB"])
def test_invalid_signature_raises_value_error(signature):
    with pytest.raises(ValueError, match="signature"):
        Calibration(signature)


def test_valid_signature():
    res = parse_calibration_signature("uK_RJ -> pW")
    assert res["in"]["units"] == "uK_RJ"
    assert res["out"]["units"] == "pW"


@pytest.mark.parametrize("band_name", ["act/pa5/f150", "atlast/f850"])
@pytest.mark.parametrize("with_spectrum", [False, True])
@pytest.mark.parametrize(
    "T_b",
    [[3.0, 30.0, 300.0], [0.0, T_CMB, 1000.0], [0.0, 1e-14, T_CMB, 300.0], [1e-5, 1.0, 3.5, 1e6], [0.0, 2e-4]],
)
def test_brightness_temperature_to_power_is_planck(band_name, with_spectrum, T_b):
    """
    K_b -> pW must follow the (non-linear) Planck response over the whole range of temperatures
    (it used to be the tangent line at the minimum temperature: 0.64x at 300 K for [3, 30, 300] K), also from 0 K
    (a linear grid from 0 to 1000 K gave 1.7x the power at T_CMB at 150 GHz), from a tiny temperature (a geometric
    grid from 1e-14 K gave 1.6x the power at T_CMB at 850 GHz), and over many decades (interpolated linearly in P
    on a geometric grid from 1e-5 K, the power at 1 K was 11% too large at 850 GHz), and from 0 K to just above
    eps, below the 1e-3 K floor of the grid (which must stay ascending).
    """
    band = maria.get_band(band_name)
    kwargs = {}
    if with_spectrum:
        spectrum = AtmosphericSpectrum(region="chajnantor")
        kwargs = {
            "spectrum": spectrum,
            "pwv": 1.0,
            "effective_temperature": spectrum.side_effective_temperature.K.mean(),
            "elevation": np.radians(60),
        }

    T_b = np.array(T_b)
    P = np.ravel(Calibration("K_b -> pW", band=band, **kwargs)(T_b))
    if with_spectrum:
        # the exact (non-interpolated) power at each temperature
        truth = 1e12 * np.array(
            [np.ravel(brightness_temperature_to_power_explicit(np.array([T]), band=band, **kwargs))[0] for T in T_b]
        )
    else:
        # P = k_B * integral of passband(nu) * T_RJ(nu) over nu, with T_RJ = (h nu / k_B) / (exp(h nu / k_B T_b) - 1)
        nu = band.nu.Hz
        with np.errstate(divide="ignore", over="ignore"):
            T_RJ = (h * nu / k_B) / np.expm1(h * nu / (k_B * T_b[:, None]))
        truth = 1e12 * k_B * np.trapezoid(band.passband(nu) * T_RJ, x=nu, axis=-1)

    assert np.allclose(P, truth, rtol=1e-3, atol=0)


@pytest.mark.parametrize(
    "T_b",
    [
        [0.0],
        [0.0, 0.0, 0.0],
        [1e-5, 2e-5],
        [0.0, 1e-5],
        [T_CMB],
        [T_CMB, T_CMB + 1e-5],
        [1e-3],
        [T_CMB, T_CMB + 6e-5],
        [T_CMB, T_CMB + 9e-5],
        [1e-3, 1e-3 + 9e-5],
    ],
)
@pytest.mark.parametrize("with_spectrum", [False, True])
def test_brightness_temperature_to_power_narrow_range(T_b, with_spectrum):
    """
    For a range of temperatures narrower than eps, the Planck power is interpolated on a grid of width eps from
    T - eps / 2, which dipped below 0 K near 0 K: the power of 0 K was -0.79 pW at 150 GHz instead of 0. The grid
    must reach max(T_b): for spreads between eps / 2 and eps the top values fell outside it, and were clamped (the
    step from T_CMB to T_CMB + 9e-5 K came out 45% too small) or raised with an atmospheric spectrum.
    """
    band = maria.get_band("act/pa5/f150")
    kwargs = {}
    if with_spectrum:
        spectrum = AtmosphericSpectrum(region="chajnantor")
        kwargs = {
            "spectrum": spectrum,
            "pwv": 1.0,
            "effective_temperature": spectrum.side_effective_temperature.K.mean(),
            "elevation": np.radians(60),
        }
    T_b = np.array(T_b)
    P = np.ravel(Calibration("K_b -> pW", band=band, **kwargs)(T_b))

    if with_spectrum:
        truth = 1e12 * np.array(
            [np.ravel(brightness_temperature_to_power_explicit(np.array([T]), band=band, **kwargs))[0] for T in T_b]
        )
    else:
        nu = band.nu.Hz
        with np.errstate(divide="ignore", over="ignore"):
            T_RJ = (h * nu / k_B) / np.expm1(h * nu / (k_B * T_b[:, None]))
        truth = 1e12 * k_B * np.trapezoid(band.passband(nu) * T_RJ, x=nu, axis=-1)

    assert np.all(P >= 0)
    # 1e-5 of the power at T_CMB, about 1e-6 pW
    assert np.allclose(P, truth, rtol=1e-3, atol=1e-6)
    # the signal steps, about 1e-5 pW for 1e-4 K at T_CMB
    assert np.allclose(P - P[0], truth - truth[0], rtol=1e-2, atol=1e-9)


@pytest.mark.parametrize("with_spectrum", [False, True])
def test_brightness_temperature_to_power_shape_and_non_finite(with_spectrum):
    """
    The power has the shape of T_b (with a spectrum a scalar gave shape (1,)), and a NaN or infinite temperature
    gives NaN for that element only. On master, a NaN made the whole conversion NaN, an infinite temperature gave
    an infinite power without a spectrum, and with a spectrum the conversion raised (the grid was not ascending).
    """
    band = maria.get_band("act/pa5/f150")
    kwargs = {}
    if with_spectrum:
        spectrum = AtmosphericSpectrum(region="chajnantor")
        kwargs = {
            "spectrum": spectrum,
            "pwv": 1.0,
            "effective_temperature": spectrum.side_effective_temperature.K.mean(),
            "elevation": np.radians(60),
        }
    convert = Calibration("K_b -> pW", band=band, **kwargs)
    truth = 1e12 * np.ravel(brightness_temperature_to_power_explicit(np.array([3.0, 30.0]), band=band, **kwargs))

    assert np.shape(convert(3.0)) == ()
    assert np.isclose(convert(3.0), truth[0], rtol=1e-6)
    assert convert(np.array([[3.0, 30.0]])).shape == (1, 2)

    P = convert(np.array([3.0, np.inf, 30.0]))
    assert np.isnan(P[1])
    assert np.allclose(P[[0, 2]], truth, rtol=1e-6)

    assert np.all(np.isnan(convert(np.array([np.nan]))))
    P = convert(np.array([3.0, np.nan, 30.0]))
    assert np.isnan(P[1])
    assert np.allclose(P[[0, 2]], truth, rtol=1e-6)


def test_brightness_temperature_to_power_spectrum_bounded_memory():
    """
    With an atmospheric spectrum, K_b -> pW integrates over frequency at each of the 256 grid temperatures.
    Broadcasting the opacity cube over them took a 4.5 GB temporary for a whole spectrum (13.6 GB peak for any
    spread of temperatures, against 0.2 GB on master); the frequency axis is now contracted with trapezoid weights.
    Checked first on a 2x2x2 slice of the spectrum (where the broadcast temporaries are a few hundred MB), against
    the integral at a node of the slice, then end to end.
    """
    band = maria.get_band("act/pa5/f150")
    spectrum = AtmosphericSpectrum(region="chajnantor")
    nu = spectrum.side_nu.Hz
    opacity = spectrum.fields["opacity"][:2, :2, :2]
    piece = SimpleNamespace(side_nu=spectrum.side_nu, fields={"opacity": opacity}, points=[p[:2] for p in spectrum.points])
    # at the (T_eff, pwv, airmass) node (0, 1, 0) of the slice
    node = {
        "effective_temperature": piece.points[0][0],
        "pwv": piece.points[1][1],
        "elevation": np.arcsin(1 / piece.points[2][0]),
    }
    T_b = np.geomspace(1.0, 300.0, 1000)

    tracemalloc.start()
    P = brightness_temperature_to_power(T_b, band, spectrum=piece, **node)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 20e6

    for T in T_b[[0, 500, 999]]:
        T_RJ = inverse_rayleigh_jeans_spectrum(planck_spectrum(T_b=T, nu=nu), nu=nu)
        truth = k_B * np.trapezoid(T_RJ * np.exp(-opacity[0, 1, 0]) * band.passband(nu), x=nu)
        assert np.isclose(P[T_b == T][0], truth, rtol=1e-3, atol=0)

    n = 100_000
    kwargs = {
        "spectrum": spectrum,
        "pwv": np.full(n, 1.0),
        "effective_temperature": np.full(n, spectrum.side_effective_temperature.K.mean()),
        "elevation": np.full(n, np.radians(60)),
    }
    T = np.random.default_rng(0).uniform(2.7, 300, size=n)
    tracemalloc.start()
    P = Calibration("K_b -> pW", band=band, **kwargs)(T)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 1e9
    scalar_kwargs = {k: v[0] for k, v in kwargs.items() if k != "spectrum"}
    truth = 1e12 * np.ravel(brightness_temperature_to_power_explicit(T[:3], band=band, spectrum=spectrum, **scalar_kwargs))
    assert np.allclose(P[:3], truth, rtol=1e-3, atol=0)


@pytest.mark.parametrize("center", [2e6, 56e6, 1e9])
def test_brightness_temperature_to_power_sub_ghz_band(center):
    """
    Below the grid floor the power is interpolated linearly. A fixed 1 mK floor is not negligible below about
    1 GHz (h nu / k_B is 2.7 mK at 56 MHz): at 56 MHz the power at 0.5 mK was 7.7x the Planck value. The floor is
    now at most 2% of h nu / k_B at the lowest frequency of the band; bands down to 1 MHz are allowed.
    """
    band = maria.Band(center=center, width=0.1 * center, shape="gaussian")
    T_b = np.array([0.0, 1e-5, 1e-4, 5e-4, 9e-4, 1.0])
    P = np.ravel(Calibration("K_b -> pW", band=band)(T_b))
    truth = 1e12 * np.array([np.ravel(brightness_temperature_to_power_explicit(np.array([T]), band=band))[0] for T in T_b])
    # the Wien-suppressed power far below h nu / k_B is negligible (and linear below the floor)
    assert np.allclose(P, truth, rtol=1e-3, atol=1e-9 * truth.max())
