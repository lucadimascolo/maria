from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.array import Array
from maria.array.generation import generate_2d_pattern
from maria.utils import compute_diameter


@pytest.mark.parametrize("band", ["act/pa5/f150", {"center": 150e9, "width": 30e9}])
def test_singular_band_key(band):
    array = Array.from_config({"n": 10, "primary_size": 5, "field_of_view": 0.2, "band": band})
    assert len(array.bands) == 1
    assert np.all(array.dets.band_name == array.bands[0].name)
    assert np.isclose(array.bands[0].center.Hz, 150e9, rtol=1e-2)


def test_array_list_subset():
    instrument = maria.get_instrument("test/1deg")
    subset = instrument.arrays.subset(band_name="f150")
    assert subset.n == (instrument.dets.band_name == "f150").sum()
    assert np.all(subset.dets.band_name == "f150")


@pytest.mark.parametrize("degrees", [True, False])
def test_focal_plane_offset_units(degrees):
    config = {"n": 1, "primary_size": 5, "field_of_view": 0.0, "bands": ["act/pa5/f150"], "degrees": degrees}
    offset = (0.01, -0.02)
    centered = Array.from_config(config)
    shifted = Array.from_config({**config, "focal_plane_offset": offset})
    expected = np.radians(offset) if degrees else np.array(offset)
    assert np.allclose(shifted.offsets - centered.offsets, expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_col": 4, "n_row": 4, "max_diameter": 1.0, "packing": "square"},
        {"n_col": 4, "n_row": 6, "max_diameter": 1.0, "packing": "triangular"},
        {"n": 16, "max_diameter": 1.0},
        {"max_diameter": 1.0, "spacing": 0.1},
    ],
)
def test_center_is_applied_with_max_diameter(kwargs):
    X = generate_2d_pattern(**kwargs, center=True)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-12)
    if "n" in kwargs or "n_col" in kwargs:
        assert np.isclose(compute_diameter(X), kwargs["max_diameter"])


def test_default_pattern_is_not_recentered():
    # the even-grid offset keeps a detector on the boresight, and is kept by default
    X = generate_2d_pattern(n_col=4, n_row=4, max_diameter=1.0, packing="square")
    assert np.isclose(np.abs(X).sum(axis=-1).min(), 0)
