from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.cmb import generate_cmb_patch, get_cmb_spectrum


def test_cmb_patch():
    plan = maria.Plan.generate(
        start_time="2025-01-01",
        scan_type="daisy",
        scan_parameters={"radius": 5, "speed": 1},  # in degrees
        duration=120,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=(0, -50),
        jitter=0,
        frame="ra/dec",
    )

    cmb_patch = maria.cmb.generate_cmb_patch(width=15, center=plan.center())

    sim = maria.Simulation(
        instrument="test/1deg",
        plans=plan,
        site="cerro_toco",
        map=cmb_patch,
        noise=False,
    )

    tod = sim.run()[0].to("K_CMB")

    # should be around 110 uK
    cmb_rms_K = tod.signal.std(axis=-1).compute()

    assert all(cmb_rms_K > 0.5 * cmb_patch.rms)
    assert all(cmb_rms_K < 2.0 * cmb_patch.rms)


@pytest.mark.parametrize("width, height", [(4.0, 2.0), (2.0, 3.0), (3.0, 3.0)])
def test_cmb_patch_geometry(width, height):
    resolution = 0.02
    m = generate_cmb_patch(width=width, height=height, resolution=resolution, buffer=1)
    nx, ny = int(width / resolution), int(height / resolution)
    assert m.data.shape[-2:] == (ny, nx)
    # the map resolution is the one the patch was generated with
    assert np.isclose(abs(m.xi_res.deg), width / nx, rtol=1e-6)
    assert np.isclose(abs(m.eta_res.deg), height / ny, rtol=1e-6)


def test_cmb_patch_spectrum():
    """The flat-sky power spectrum of generated patches matches the lensed (Planck) TT spectrum, also at high ell."""
    cl = get_cmb_spectrum(source="lensed")
    bins = [(500, 1500), (1500, 2500), (3000, 4500)]
    ratios = {b: [] for b in bins}

    np.random.seed(0)
    for _ in range(8):
        # buffer=0 makes the patch periodic, so its FFT has no edge leakage
        m = generate_cmb_patch(width=5.0, resolution=0.02, buffer=0)
        d = np.asarray(m.data).squeeze().astype(float)
        ny, nx = d.shape
        dx, dy = abs(m.xi_res.rad), abs(m.eta_res.rad)
        P = np.abs(np.fft.fft2(d)) ** 2 * dx * dy / (nx * ny)
        KX, KY = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy))
        ell = 2 * np.pi * np.hypot(KX, KY)
        for b in bins:
            in_bin = (ell >= b[0]) & (ell < b[1])
            ratios[b].append(P[in_bin].sum() / np.interp(ell[in_bin], cl.ell, cl.TT).sum())

    for b in bins:
        assert np.isclose(np.mean(ratios[b]), 1.0, atol=0.05), (b, np.mean(ratios[b]))
