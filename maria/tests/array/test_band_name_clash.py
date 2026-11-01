from __future__ import annotations

import maria
import numpy as np
from maria import Simulation
from maria.array import Array
from maria.plan import get_plan


def test_same_named_bands_keep_their_own_noise():
    """
    Two arrays that define different bands with the same name must keep their own passband and noise level,
    and the combined detectors must remember which array they come from.
    """
    arrays = {
        "A": {
            "n": 20,
            "primary_size": 5,
            "field_of_view": 0.2,
            "bands": {"f150": {"center": 150e9, "width": 10e9, "NET_RJ": 100e-6, "knee": 0.0}},
        },
        "B": {
            "n": 20,
            "primary_size": 5,
            "field_of_view": 0.2,
            "bands": {"f150": {"center": 150e9, "width": 40e9, "NET_RJ": 1000e-6, "knee": 0.0}},
        },
    }

    instrument = maria.Instrument(arrays=arrays)

    assert sorted(np.unique(instrument.dets.array_name)) == ["A", "B"]
    assert len(instrument.bands) == 2

    widths = {band.name: band.width.Hz for band in instrument.bands}
    for array_name, width in [("A", 10e9), ("B", 40e9)]:
        band_names = np.unique(instrument.dets.band_name[instrument.dets.array_name == array_name])
        assert len(band_names) == 1
        assert np.isclose(widths[band_names[0]], width, rtol=1e-2)

    sample_rate = 50.0
    plan = get_plan("ten_second_zenith_stare", duration=60, sample_rate=sample_rate)
    np.random.seed(0)
    sim = Simulation(instrument=instrument, plans=plan, site="llano_de_chajnantor", progress_bars=False)
    tod = sim.run(units="K_RJ")[0]
    noise = np.asarray(tod.data["noise"], dtype=float)

    # per-sample sigma = NET * sqrt(sample_rate)
    measured_net = noise.std(axis=-1) / np.sqrt(sample_rate)
    for array_name, net in [("A", 100e-6), ("B", 1000e-6)]:
        mask = tod.dets.array_name == array_name
        ratio = measured_net[mask].mean() / net
        # 20 dets x 3000 samples: the relative error on the mean NET is about 0.3%
        assert np.abs(ratio - 1) < 0.02, f"array {array_name}: measured NET / requested = {ratio:.3f}"


def test_identical_bands_are_not_renamed():
    instrument = maria.get_instrument("so/sat")
    assert instrument.bands.names == [instrument.dets.band_name[0]]
    assert len(np.unique(instrument.dets.array_name)) == 7


def test_renaming_does_not_change_the_callers_arrays():
    band = {"center": 150e9, "width": 10e9, "NET_RJ": 100e-6}
    a = Array.from_config({"name": "A", "n": 4, "primary_size": 5, "field_of_view": 0.2, "bands": {"f150": band}})
    b = Array.from_config(
        {"name": "B", "n": 4, "primary_size": 5, "field_of_view": 0.2, "bands": {"f150": {**band, "width": 40e9}}}
    )
    instrument = maria.Instrument(arrays=[a, b])
    assert sorted(instrument.bands.names) == ["A/f150", "B/f150"]
    for array in [a, b]:
        assert array.bands.names == ["f150"]
        assert np.all(array.dets.band_name == "f150")
