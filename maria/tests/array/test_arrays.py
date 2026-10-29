from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest

plt.close("all")


arrays1 = {
    "subarray-1": {
        "n": 500,
        "primary_size": 10,
        "field_of_view": 2,
        "shape": "hexagon",
        "bands": [{"center": 30e9, "width": 5e9}, {"center": 40e9, "width": 5e9}],
    },
    "subarray-2": {
        "n": 500,
        "primary_size": 10,
        "field_of_view": 2,
        "shape": "hexagon",
        "bands": [{"center": 90e9, "width": 5e9}, {"center": 150e9, "width": 5e9}],
        "rotation": 90,
    },
}

# /Users/tom/maria/src/maria/array/

arrays2 = [{"file": "data/alma/alma_f144.csv", "primary_size": 12, "bands": ["alma/f144"]}]


offsets = np.random.uniform(low=0, high=1, size=(100, 2))
arrays3 = [{"bands": ["act/pa4/f220"], "primary_size": 5, "xi": offsets[..., 0], "eta": offsets[..., 1], "degrees": True}]

array_configs_to_test = [arrays1, arrays2, arrays3]


@pytest.mark.parametrize("array_config", array_configs_to_test)
def test_generate_array(array_config):
    instrument = maria.Instrument(arrays=array_config)


def test_plot_gammas_follow_the_stokes_weights():
    """plot(plot_gammas=True) draws each detector along its polarization: gamma = 0 couples to +Q (north-south,
    along eta at transit) and gamma = 45 deg to +U (towards -xi), so the line is along (-sin gamma, cos gamma)."""
    instrument = maria.get_instrument(
        array={"n": 10, "field_of_view": 0.1, "bands": ["act/pa5/f150"], "primary_size": 6, "polarized": True}
    )
    dets = instrument.dets
    dets.plot(plot_gammas=True)
    lines = [line for line in plt.gcf().axes[0].get_lines() if len(line.get_xdata()) == 2]
    x = np.stack([line.get_xdata() for line in lines], axis=-1)
    y = np.stack([line.get_ydata() for line in lines], axis=-1)
    assert x.shape == (2, dets.n)
    direction = np.stack([x[1] - x[0], y[1] - y[0]])
    direction /= np.linalg.norm(direction, axis=0)
    expected = np.stack([-np.sin(dets.gamma), np.cos(dets.gamma)])
    assert np.allclose(np.abs(np.sum(direction * expected, axis=0)), 1)
    plt.close("all")
