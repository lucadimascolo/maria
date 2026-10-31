from __future__ import annotations

import maria
import numpy as np
import pytest
from maria import Band


def test_band_center_on_non_uniform_grid():
    # a flat 100-200 GHz band sampled more densely at low frequencies
    nu = np.sort(np.r_[np.linspace(100e9, 120e9, 200), np.linspace(120e9, 200e9, 50)[1:]])
    band = Band(nu=nu, tau=np.ones(len(nu)), name="flat")
    assert np.isclose(band.center.Hz, 150e9, rtol=1e-6)


def test_auto_band_centers_unchanged():
    for name in ["act/pa5/f150", "so/f093", "atlast/f350"]:
        band = maria.get_band(name)
        assert np.isclose(band.center.Hz, np.sum(band.nu.Hz * band.tau) / np.sum(band.tau), rtol=1e-6)


@pytest.mark.parametrize(
    "center, name",
    [
        (30e9, "f030"),
        (150e9, "f150"),
        (999e9, "f999"),
        (1e12, "f1000"),
        (1.5e12, "f1500"),
        (2.5e12, "f2500"),
        # below 1 GHz the names stay in MHz, as on master
        (150e6, "f150"),
        (56e6, "f056"),
    ],
)
def test_auto_band_names(center, name):
    assert Band(center=center, width=0.1 * center).name == name


def test_time_constant_warns(caplog):
    Band(center=150e9, width=30e9, NET_RJ=1e-4, time_constant=0.0)
    assert "time constant" not in caplog.text
    Band(center=150e9, width=30e9, NET_RJ=1e-4, time_constant=0.1)
    assert "time constant" in caplog.text
