from __future__ import annotations

import maria
import numpy as np
import pytest
from maria import Simulation
from maria.plan import get_plan

SAMPLE_RATE = 50.0


def measure_net(spec: dict, polarized: bool):
    """
    The white noise level (in K_RJ√s) of a noise-only simulation of 20 (or 2 x 20 polarized) detectors.
    """
    band = {"center": 150e9, "width": 30e9, "knee": 0.0, **spec}
    array = {"n": 20, "primary_size": 5, "field_of_view": 0.2, "polarized": polarized, "bands": {"f150": band}}
    instrument = maria.Instrument(arrays=[array])
    plan = get_plan("ten_second_zenith_stare", duration=60, sample_rate=SAMPLE_RATE)
    np.random.seed(0)
    sim = Simulation(instrument=instrument, plans=plan, site="llano_de_chajnantor", progress_bars=False)
    tod = sim.run(units="K_RJ")[0]
    noise = np.asarray(tod.data["noise"], dtype=float)
    # the per-sample standard deviation is NET * sqrt(sample_rate)
    return (noise.std(axis=-1) / np.sqrt(SAMPLE_RATE)).mean(), instrument.bands[0]


@pytest.mark.parametrize("polarized", [False, True])
def test_requested_net_is_the_detector_white_noise_level(polarized):
    """
    The TOD of a polarized detector in K_RJ is referred to Stokes I (a uniform 1 K_RJ map reads 1 K_RJ), so a
    requested NET must be the white noise level of every detector, polarized or not (it used to be 2x for
    polarized detectors).
    """
    measured, band = measure_net({"NET_RJ": 100e-6}, polarized=polarized)
    # 20-40 dets x 3000 samples: the relative error is about 0.3%
    assert np.abs(measured / 100e-6 - 1) < 0.02, f"polarized={polarized}: measured / requested = {measured / 100e-6:.3f}"
    assert np.isclose(band.NET_RJ.to("K_RJ√s"), 100e-6)

    measured, band = measure_net({"NET_CMB": 100e-6}, polarized=polarized)
    expected = band.NET_RJ.to("K_RJ√s")
    assert np.abs(measured / expected - 1) < 0.02, f"polarized={polarized}: measured / requested = {measured / expected:.3f}"
    assert np.isclose(band.NET_CMB.to("K_CMB√s"), 100e-6)


@pytest.mark.parametrize("polarized", [False, True])
def test_nep_specified_band_gives_polarized_detectors_twice_the_net(polarized):
    """
    An NEP is the same for every detector, so a polarized detector's NET is twice that of an unpolarized one.
    """
    measured, band = measure_net({"NEP": 1e-17}, polarized=polarized)
    expected = (2.0 if polarized else 1.0) * band.NET_RJ.to("K_RJ√s")
    assert np.abs(measured / expected - 1) < 0.02
    assert np.isclose(band.NEP.to("W√s"), 1e-17)
