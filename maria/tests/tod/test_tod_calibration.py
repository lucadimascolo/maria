from __future__ import annotations

import maria
import numpy as np
import pytest
from maria import Simulation
from maria.constants import k_B
from maria.plan import get_plan


@pytest.mark.parametrize("units", ["K_RJ", "K_CMB"])
def test_mixed_polarization_band_calibration(units):
    """
    In a band shared by polarized and unpolarized detectors, each detector must be calibrated with its own
    polarization (the unpolarized ones used to be calibrated as polarized, reading 2x).
    """
    arrays = {
        "P": {"n": 10, "primary_size": 5, "field_of_view": 0.2, "polarized": True, "bands": ["act/pa5/f150"]},
        "U": {"n": 10, "primary_size": 5, "field_of_view": 0.2, "polarized": False, "bands": ["act/pa5/f150"]},
    }
    instrument = maria.Instrument(arrays=arrays)
    plan = get_plan("ten_second_zenith_stare", duration=10, sample_rate=20)
    sim = Simulation(instrument=instrument, plans=plan, site="llano_de_chajnantor", progress_bars=False)
    tod_pW = sim.run(units="pW")[0]
    tod = tod_pW.to(units)

    band = instrument.bands[0]
    polarized = ~np.isnan(tod.dets.gamma)
    assert polarized.any() and (~polarized).any()

    ratio = np.asarray(tod.data["noise"], dtype=float) / np.asarray(tod_pW.data["noise"], dtype=float)
    per_det_ratio = np.median(ratio, axis=-1)

    if units == "K_RJ":
        expected_unpolarized = 1 / (1e12 * k_B * band.compute_transmission_integral())
    else:
        expected_unpolarized = band.cal("pW -> K_CMB", polarized=False)(1.0)

    assert np.allclose(per_det_ratio[~polarized], expected_unpolarized, rtol=1e-4)
    assert np.allclose(per_det_ratio[polarized], 2 * expected_unpolarized, rtol=1e-4)


def test_calibration_kwargs_without_band():
    plan = get_plan("ten_second_zenith_stare", duration=5, sample_rate=10)
    sim = Simulation(instrument="test/1deg", plans=plan, site="llano_de_chajnantor", progress_bars=False)
    tod = sim.run()[0]

    kwargs = tod.calibration_kwargs()
    assert kwargs["elevation"].shape == tod.shape
    assert kwargs["polarized"].shape == (tod.dets.n, 1)

    band = tod.dets.bands[0]
    band_kwargs = tod.calibration_kwargs(band)
    n_band = (tod.dets.band_name == band.name).sum()
    assert band_kwargs["elevation"].shape == (n_band, tod.shape[-1])
    assert np.array_equal(band_kwargs["polarized"][:, 0], ~np.isnan(tod.dets.gamma[tod.dets.band_name == band.name]))
