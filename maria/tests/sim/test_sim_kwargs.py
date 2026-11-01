from __future__ import annotations

import healpy as hp
import maria
import numpy as np
import pytest
from maria.map import HEALPixMap, ProjectionMap


def sim(**kwargs):
    plan = maria.Plan.generate(
        start_time="2025-01-01T07:00:00", duration=2, sample_rate=10, scan_center=(150.0, -30.0), frame="ra/dec"
    )
    return maria.Simulation(
        instrument="test/1deg", plans=plan, site="cerro_toco", noise=False, progress_bars=False, **kwargs
    )


@pytest.mark.parametrize("cmb_kwargs", [{"nsides": 16}, {"seeds": 1}])
def test_unknown_cmb_kwargs_raise(cmb_kwargs):
    with pytest.raises(ValueError, match="Invalid cmb_kwargs"):
        sim(cmb="generate", cmb_kwargs={"nside": 16, **cmb_kwargs})


def test_unknown_map_kwargs_raise():
    m = ProjectionMap(data=np.zeros((10, 10)), width=0.5, nu=150e9, center=(150.0, -30.0), units="K_RJ")
    with pytest.raises(ValueError, match="Invalid map_kwargs"):
        sim(map=m, map_kwargs={"bilinear": True})


def test_valid_kwargs():
    sim(cmb="generate", cmb_kwargs={"nside": 16, "lmax": 32, "seed": 1})
    m = HEALPixMap(data=np.zeros(hp.nside2npix(16)), units="K_RJ")
    sim(map=m, map_kwargs={"bilinear_sampling": True})


def test_map_name_with_simulation_kwargs():
    """With a map name, the simulation kwargs are not passed to the map loader (master raised a TypeError)."""
    sim(map="maps/cluster1.fits", map_kwargs={"bilinear_sampling": True, "nu": 150e9})
    with pytest.raises(TypeError):
        sim(map="maps/cluster1.fits", map_kwargs={"nu": 150e9, "not_a_map_kwarg": 1})
