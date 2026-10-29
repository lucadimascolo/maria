from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.beam import compute_angular_fwhm
from maria.constants import c, k_B
from maria.coords import Coordinates, offsets_to_phi_theta
from maria.instrument import Band
from maria.map import ProjectionMap
from scipy.special import erf

CENTER = (150.0, -30.0)


def test_samples_off_the_map(simulate_map):
    # a 0.5 deg map that is zero inside a constant border
    data = np.zeros((50, 50))
    data[[0, -1], :] = data[:, [0, -1]] = 1e-2
    m = ProjectionMap(data=data, width=0.5, nu=150e9, center=CENTER, units="K_RJ")

    def run(center):
        # a 50 m dish, whose beam barely smooths the border
        sim = simulate_map(m, primary_size=50, center=center, radius=0.1)
        return sim.run(units="K_RJ")[0].signal.compute()

    # the sky outside the input map is zero (it used to be the value of the nearest edge pixel, 1e-2)
    off_center = (CENTER[0] + 3.0, CENTER[1])
    assert np.all(run(off_center) == 0)

    # on the map, the samples see the map
    assert np.isfinite(run(CENTER)).all()


@pytest.mark.parametrize("row, col", [(50, 50), (50, 0), (0, 0)])
def test_point_source_at_the_edge_of_the_map(simulate_map, row, col):
    """A point source of flux S peaks at S / beam_area wherever it is on the map: the edges do not reflect the sky."""
    nu, primary_size, flux = 150e9, 10, 1.0
    fwhm = compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=nu)
    data = np.zeros((101, 101))
    data[row, col] = flux
    m = ProjectionMap(data=data, resolution=np.degrees(fwhm) / 10, nu=nu, center=CENTER, units="Jy/pixel")
    center = np.degrees(offsets_to_phi_theta(np.array([m.xi.rad[col], m.eta.rad[row]]), m.center[0].rad, m.center[1].rad))

    signal = simulate_map(m, primary_size=primary_size, center=tuple(center), radius=1e-4, duration=10)
    band_nu = np.linspace(145e9, 155e9, 1001)
    expected = np.mean(flux * 1e-26 * c**2 / (2 * k_B * band_nu**2) / (np.pi / (4 * np.log(2)) * fwhm**2))
    assert np.isclose(signal.run(units="K_RJ")[0].signal.compute().mean(), expected, rtol=0.01)


@pytest.mark.parametrize("weight", [None, 3.0])
@pytest.mark.parametrize("offset", [-0.3, 0.1, 0.3, 0.5, 1.5])
def test_beam_of_a_source_at_the_edge_reaches_past_the_map(simulate_map, offset, weight):
    """The sky is the map with zero sky beyond its edges, convolved with the beam: a point source in the edge pixel
    is seen a fraction f of a FWHM from it at S / beam_area * exp(-4 ln2 f^2), inside or outside the map, out to the
    4 sigma of the smoothing kernel (1.5 FWHM is 3.5 sigma), with or without a (constant) weight (the samples beyond
    the edge used to be zero)."""
    nu, primary_size, flux = 150e9, 10, 1.0
    fwhm = compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=nu)
    data = np.zeros((101, 101))
    data[50, 100] = flux
    weight = None if weight is None else np.full(data.shape, weight)
    m = ProjectionMap(data=data, weight=weight, resolution=np.degrees(fwhm) / 10, nu=nu, center=CENTER, units="Jy/pixel")
    xi = m.xi.rad[100] + offset * fwhm
    center = np.degrees(offsets_to_phi_theta(np.array([xi, m.eta.rad[50]]), m.center[0].rad, m.center[1].rad))

    signal = simulate_map(m, primary_size=primary_size, center=tuple(center), radius=1e-5, duration=5)
    band_nu = np.linspace(145e9, 155e9, 1001)
    peak = np.mean(flux * 1e-26 * c**2 / (2 * k_B * band_nu**2) / (np.pi / (4 * np.log(2)) * fwhm**2))
    expected = peak * np.exp(-4 * np.log(2) * offset**2)
    assert np.isclose(signal.run(units="K_RJ")[0].signal.compute().mean(), expected, rtol=0.01)


def test_map_smaller_than_the_beam(simulate_map):
    """A uniform map smaller than the beam, seen at its centre, gives the beam-weighted integral over the map only."""
    primary_size, width, n = 0.5, 0.2, 41
    sigma = np.degrees(compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=150e9)) / np.sqrt(8 * np.log(2))
    m = ProjectionMap(data=1e-3 * np.ones((n, n)), width=width, nu=150e9, center=CENTER, units="K_RJ")

    signal = simulate_map(m, primary_size=primary_size, radius=1e-4, duration=5).run(units="K_RJ")[0].signal.compute()
    half_width = width / 2 * n / (n - 1)  # to the outer edge of the edge pixels
    assert np.isclose(signal.mean(), 1e-3 * erf(half_width / (sigma * np.sqrt(2))) ** 2, rtol=0.01)


@pytest.mark.parametrize("weighted", [False, True])
def test_padding_only_as_far_as_the_samples_reach(weighted):
    """The map is padded with zero sky only as far as the samples reach beyond its outer pixel centres (and at most
    the kernel radius), on the sides they reach; the smoothed pixels are the same as with the full padding (a small
    map with a wide beam used to be padded by the full kernel radius on every side)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 30))
    weight = rng.uniform(0.5, 2.0, size=data.shape) if weighted else None
    m = ProjectionMap(data=data, weight=weight, resolution=0.01, center=CENTER, units="K_RJ")
    sigma, radius = np.radians(0.1), np.radians(0.395)  # 40 pixels
    # samples from the middle to 2.5 pixels beyond the last pixel centre in xi, and inside the map in eta
    offsets = np.stack([np.linspace(0, m.xi.rad[-1] + np.radians(0.025), 50), np.zeros(50)], axis=-1)
    padded = m._pad_with_zero_sky(radius=radius, offsets=offsets)
    full = m._pad_with_zero_sky(radius=radius)
    assert padded.data.shape[-2:] == (20, 34)
    assert full.data.shape[-2:] == (100, 110)
    smoothed = padded._smooth(sigma=sigma, degrees=False, zero_outside=True).data.compute()
    smoothed_full = full._smooth(sigma=sigma, degrees=False, zero_outside=True).data.compute()
    assert np.array_equal(smoothed, smoothed_full[..., 40:60, 40:74])
    assert np.allclose(padded.xi.rad, full.xi.rad[40:74]) and np.allclose(padded.eta.rad, full.eta.rad[40:60])


@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_pointing_matrix_outside_the_map(bilinear, descending):
    """Samples beyond the outer pixel edges get zero weight; the others (and the default) keep theirs."""
    resolution = 0.1  # deg; 5 pixels, so the outer pixel edges are at +/- 0.25 deg
    m = ProjectionMap(data=np.ones((5, 5)), resolution=resolution, center=CENTER, units="K_RJ")
    if descending:
        m = m[..., ::-1]
    xi = np.radians([-0.26, -0.24, 0.0, 0.24, 0.26])
    phi_theta = offsets_to_phi_theta(np.stack([xi, np.zeros_like(xi)], axis=-1), *np.radians(CENTER))
    coords = Coordinates(phi=phi_theta[:, None, 0], theta=phi_theta[:, None, 1], t=1.7e9 * np.ones(1), frame="ra/dec")

    def weights(**kwargs):
        return np.asarray(m._pointing_matrix_ingredients(coords, bilinear=bilinear, **kwargs)[2]).sum(axis=0).ravel()

    assert np.allclose(weights(), 1)
    assert np.allclose(weights(zero_outside=True), [0, 1, 1, 1, 0])


@pytest.mark.parametrize("bilinear", [False, True])
def test_pointing_matrix_inside_the_map_is_unchanged(simulate_map, bilinear):
    """For a scan inside a polarized map, the masked pointing matrix of the simulation is bit-identical to the
    public one (the mask used to make its weights float32, so Q and U changed at float32 rounding)."""
    rng = np.random.default_rng(0)
    m = ProjectionMap(data=rng.standard_normal((3, 50, 50)), stokes="IQU", width=1.0, center=CENTER, units="K_RJ")
    band = Band(center=150e9, width=10e9, NET_RJ=1e-5)
    instrument = maria.get_instrument(
        array={"n": 20, "field_of_view": 0.1, "bands": [band], "primary_size": 10, "polarized": True}
    )
    obs = simulate_map(m, instrument=instrument, radius=0.1, duration=5).obs_list[0]
    kwargs = {"coords": obs.coords, "dets": obs.instrument.dets, "bilinear": bilinear}
    P = m.stokes_weighted_pointing_matrix(**kwargs)
    P_sim = m._stokes_weighted_pointing_matrix(zero_outside=True, **kwargs)
    assert P_sim.dtype == P.dtype
    x = m.data.compute().ravel()
    assert np.array_equal(P_sim @ x, P @ x)
