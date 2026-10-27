from __future__ import annotations

import healpy as hp
import maria
import numpy as np
import pytest
from maria.cmb import generate_cmb, get_cmb_spectrum


@pytest.mark.parametrize("nside", [256, 512, 1024])
def test_generate_cmb(nside):
    cmb = maria.cmb.generate_cmb(nside=nside)
    # cmb.plot()


def test_cmb_calibration():
    plan = maria.Plan.generate(
        scan_type="daisy",
        scan_parameters={"az_center": 150, "el_center": 50, "radius": 10, "speed": 1},  # in degrees
        duration=120,  # in seconds
        sample_rate=50,  # in Hz
        jitter=0,
    )

    sim = maria.Simulation(
        instrument="test/1deg",
        plans=plan,
        site="cerro_toco",
        cmb="generate",
        cmb_kwargs={"nside": 512},  # the beam smoothing of a full-sky map is slow at high nside
        noise=False,
    )

    tod = sim.run()[0].to("uK_CMB")

    # should be around 110 uK
    cmb_rms_uK = tod.signal.std(axis=-1).compute()

    assert all(cmb_rms_uK > 80)
    assert all(cmb_rms_uK < 150)


def test_generate_cmb_keeps_global_rng():
    """generate_cmb must not reseed the global NumPy RNG, so later draws stay independent."""
    np.random.seed(1)
    expected = np.random.standard_normal(8)

    np.random.seed(1)
    generate_cmb(nside=32)
    assert np.array_equal(np.random.standard_normal(8), expected)


def test_generate_cmb_seed():
    """The CMB realisation depends only on the seed, and the default one is unchanged."""
    a = np.asarray(generate_cmb(nside=32, seed=7).data)
    np.random.seed(99)
    b = np.asarray(generate_cmb(nside=32, seed=7).data)
    c = np.asarray(generate_cmb(nside=32, seed=8).data)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)

    # the default realisation is the one healpy draws after np.random.seed(123456)
    cl = get_cmb_spectrum(source="lensed")
    lmax = int(np.pi / hp.pixelfunc.nside2resol(32))
    np.random.seed(123456)
    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    reference = hp.alm2map(alm, nside=32, lmax=lmax)
    assert np.array_equal(np.asarray(generate_cmb(nside=32).data).reshape(3, -1), reference.astype(np.float32))
