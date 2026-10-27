from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.coords import Coordinates, offsets_to_phi_theta
from maria.instrument import Band
from maria.map import ProjectionMap

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
