from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.constants import c
from maria.io import fetch
from maria.map import ProjectionMap, all_maps

plt.close("all")


def test_map_extend():
    map_filename = fetch("maps/cluster1.fits")

    m1 = maria.map.load(filename=map_filename, nu=90e9)
    m2 = maria.map.load(filename=map_filename, nu=150e9)
    m3 = maria.map.load(filename=map_filename, nu=220e9)

    m4 = m1.extend([m2, m3], dim="nu").unsqueeze("stokes")
    m5, m6 = m4.copy(), m4.copy()
    m5.stokes = np.array(["Q"])
    m6.stokes = np.array(["U"])

    m4.extend([m5, m6], dim="stokes")


def test_map_slice():
    stokes = "IQUV"
    nu = [90e9, 150e9, 220e9]
    t = 1.7e9 + np.arange(0, 600, 120)
    data = np.random.standard_normal((len(stokes), len(nu), len(t), 100, 100))

    m = ProjectionMap(data=data, width=1e0, stokes=stokes, nu=nu, t=t, center=(0, -30), units="K_RJ", frame="ra_dec")

    m[0, :, ::2, :2]


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_map_reduce(map_path):

    m = maria.map.load(fetch(map_path))
    m.reduce((2, 7))


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_map_trim(map_path):

    m = maria.map.load(fetch(map_path))
    m.trim()


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_map_trim(map_path):

    m = maria.map.load(fetch(map_path))
    m.squeeze()


def test_strided_slice_keeps_flux():
    data = np.random.default_rng(0).uniform(0, 1, size=(1, 128, 128))
    m = ProjectionMap(data=data, nu=[150e9], resolution=0.01, units="Jy/pixel")
    assert np.isclose(m[..., ::2, ::2].data.sum().compute(), m.data.sum().compute(), rtol=2e-2)
    # reversing an axis does not change the pixel size
    assert np.allclose(m[..., ::-1, :].data.compute(), data[..., ::-1, :])
    # intensive units are unchanged by striding
    k = ProjectionMap(data=data, nu=[150e9], resolution=0.01, units="K_RJ")[..., ::2, ::2]
    assert np.allclose(k.data.compute(), data[..., ::2, ::2])

    # an inverse-variance weight goes as the inverse square of the data: x4 for Jy/pixel, so 1/16
    w = ProjectionMap(data=data, weight=np.ones_like(data), nu=[150e9], resolution=0.01, units="Jy/pixel")[..., ::2, ::2]
    assert np.allclose(w.data.compute(), 4 * data[..., ::2, ::2]) and np.allclose(w.weight.compute(), 1 / 16)
    k = ProjectionMap(data=data, weight=np.ones_like(data), nu=[150e9], resolution=0.01, units="K_RJ")[..., ::2, ::2]
    assert np.allclose(k.weight.compute(), 1.0)


def test_reduce_weighted_mean_frame_and_weight():
    data, weight = np.ones((1, 8, 8)), np.ones((1, 8, 8))
    data[..., ::2, ::2], weight[..., ::2, ::2] = 100.0, 0.0  # bad pixels with zero weight
    m = ProjectionMap(data=data, weight=weight, nu=[150e9], resolution=0.01, units="K_RJ", frame="galactic")
    r = m.reduce((2, 2))
    assert np.allclose(r.data.compute(), 1.0)
    assert np.allclose(r.weight.compute(), 3.0)  # three good pixels per block
    assert r.frame.name == "galactic"

    # an unweighted map reduces to the block mean, and Jy/pixel to the block sum
    d = np.random.default_rng(0).uniform(size=(1, 8, 8))
    u = ProjectionMap(data=d, nu=[150e9], resolution=0.01, units="Jy/pixel").reduce((2, 2))
    assert np.allclose(u.data.compute(), d.reshape(1, 4, 2, 4, 2).sum(axis=(2, 4)), rtol=1e-5)

    # NaN pixels do not count, also in an unweighted map
    d = np.ones((1, 8, 8))
    d[..., ::3, ::3] = np.nan
    u = ProjectionMap(data=d, nu=[150e9], resolution=0.01, units="K_RJ").reduce((2, 2))
    assert np.allclose(u.data.compute(), 1.0)


def test_resample_keeps_frame_and_weight():
    rng = np.random.default_rng(1)
    data, weight = rng.standard_normal((1, 16, 16)), rng.uniform(1, 2, size=(1, 16, 16))
    m = ProjectionMap(
        data=data, weight=weight, nu=[150e9], resolution=0.01, center=(30.0, 20.0), units="K_RJ", frame="galactic"
    )
    r = m.resample(m)
    assert r.frame.name == "galactic"
    inner = (..., slice(1, -1), slice(1, -1))
    assert np.allclose(r.data.compute()[inner], data[inner], atol=1e-2)
    assert np.allclose(r.weight.compute()[inner], weight[inner], atol=1e-2)


def test_to_obs_frame():
    data = np.ones((2, 1, 8, 8))
    m = ProjectionMap(data=data[:, 0], nu=[90e9, 150e9], resolution=0.01, units="K_RJ")
    assert np.allclose(m.to_obs_frame().nu.Hz, [90e9, 150e9])

    for kwargs in [{"z": 0.0}, {"v": [0.0]}]:
        out = ProjectionMap(data=data, nu=[90e9, 150e9], resolution=0.01, units="K_RJ", **kwargs).to_obs_frame()
        assert not {"z", "v"} & set(out.dims)
        assert np.allclose(out.nu.Hz, [90e9, 150e9])
        assert np.allclose(out.data.compute(), m.data.compute())

    # a line emitted at 230 GHz is observed at 115 GHz from z = 1, and at 230 / (1 + z) GHz from a receding source
    out = ProjectionMap(data=data, nu=[230e9, 345e9], resolution=0.01, units="K_RJ", z=1.0).to_obs_frame()
    assert np.allclose(out.nu.Hz, [115e9, 172.5e9])
    beta = 0.1
    out = ProjectionMap(data=data, nu=[230e9, 345e9], resolution=0.01, units="K_RJ", v=[beta * c]).to_obs_frame()
    assert np.allclose(out.nu.Hz, np.array([230e9, 345e9]) * np.sqrt((1 - beta) / (1 + beta)))
