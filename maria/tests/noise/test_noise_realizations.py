from __future__ import annotations

import maria
import numpy as np
import pytest
from maria import Simulation
from maria.noise import generate_noise_with_knee
from maria.plan import get_plan

NET_RJ = 100e-6  # in K_RJ√s


def noise_array(n=60, knee=1.0, field_of_view=0.2, **band_kwargs):
    band = {"center": 150e9, "width": 30e9, "NET_RJ": NET_RJ, "knee": knee, **band_kwargs}
    return {"n": n, "primary_size": 5, "field_of_view": field_of_view, "bands": {"f150": band}}


def simulate_noise(arrays, noise_kwargs={}, duration=120, sample_rate=100.0, n_plans=1, units="K_RJ", np_seed=0):
    instrument = maria.Instrument(arrays=[arrays] if isinstance(arrays, dict) and "n" in arrays else arrays)
    plans = [
        get_plan("ten_second_zenith_stare", duration=duration, sample_rate=sample_rate, start_time=1.7e9 + 3600 * i)
        for i in range(n_plans)
    ]
    np.random.seed(np_seed)
    sim = Simulation(
        instrument=instrument,
        plans=plans,
        site="llano_de_chajnantor",
        noise_kwargs=noise_kwargs,
        progress_bars=False,
    )
    tods = sim.run(units=units)
    return [np.asarray(tod.data["noise"], dtype=float) for tod in tods], tods


def whitened_correlation(a, b, sample_rate, knee, f_max=np.inf):
    """
    The correlation of each row of a and b below f_max, with the Fourier modes whitened by the noise model
    1 + knee / f so that each one has the same weight: the standard error is about 1 / sqrt(2 f_max T).
    """
    f = np.fft.rfftfreq(a.shape[-1], d=1 / sample_rate)
    m = (f > 0) & (f < f_max)
    A, B = (np.fft.rfft(x, axis=-1)[..., m] / np.sqrt(1 + knee / f[m]) for x in (a, b))
    return np.real(A * B.conj()).sum(-1) / np.sqrt((np.abs(A) ** 2).sum(-1) * (np.abs(B) ** 2).sum(-1))


def test_white_noise_level_with_default_correlated_noise():
    """
    The white noise level must be the requested NET with the default correlated noise settings
    (the correlated modes used to carry white noise as well, giving 120 instead of 100 uK√s).
    """
    sample_rate = 100.0
    (noise,), _ = simulate_noise(noise_array(knee=1.0), sample_rate=sample_rate)

    # in a band where the pink noise is small, divide out the model 1 + knee / f
    X = np.fft.rfft(noise, axis=-1)
    f = np.fft.rfftfreq(noise.shape[-1], d=1 / sample_rate)
    m = (f > 20) & (f < 45)
    two_sided_psd = np.abs(X[:, m]) ** 2 / (sample_rate * noise.shape[-1])
    measured_net = np.sqrt((two_sided_psd / (1 + 1.0 / f[m])).mean())

    # 60 dets x ~3000 modes: the relative error is about 0.1%
    assert np.abs(measured_net / NET_RJ - 1) < 0.02, f"measured NET / requested = {measured_net / NET_RJ:.3f}"


def test_correlated_noise_is_pink_only():
    """
    A fraction cp of the pink noise, and none of the white noise, must be common to the detectors of a band (the
    correlated modes used to carry white noise as well). With <= 16 detectors per band there is a single common
    mode, so the mean of n detectors has a PSD of cp P + ((1 - cp) P + 1) / n, with P = knee / f.
    """
    sample_rate, cp, knee, n = 100.0, 0.5, 1.0, 12
    (noise,), _ = simulate_noise(
        noise_array(n=n, knee=knee),
        noise_kwargs={"correlated_noise_proportion": cp},
        duration=300,
        sample_rate=sample_rate,
    )
    common = noise.mean(axis=0) / NET_RJ
    f = np.fft.rfftfreq(len(common), d=1 / sample_rate)
    psd = np.abs(np.fft.rfft(common)) ** 2 / (sample_rate * len(common))

    # each bin is chi-squared with 2 dof, so the standard error is 1 / sqrt(n_bins): 6% (300 bins) and 1% (1e4)
    for f_min, f_max, tolerance in [(0.01, 1.0, 0.25), (10.0, 45.0, 0.05)]:
        m = (f > f_min) & (f < f_max)
        model = cp * knee / f[m] + ((1 - cp) * knee / f[m] + 1) / n
        ratio = (psd[m] / model).mean()
        assert np.abs(ratio - 1) < tolerance, f"{f_min}-{f_max} Hz: common-mode PSD / model = {ratio:.3f}"


def test_modes_are_independent_of_the_detector_pink_noise():
    """
    The correlated modes must be independent of each detector's own pink noise, and the basis rows normalized, so
    that the pink level is the one set by the knee for any basis. Mode j used to be detector j's own pink noise
    (the same hard-coded key), so with basis = 3 * identity and cp = 0.5 the pink power was about 8x too high.
    """
    n, n_samples, sample_rate, knee = 32, 2**14, 20.0, 1.0
    x = generate_noise_with_knee(
        shape=(n, n_samples), sample_rate=sample_rate, knee=knee, basis=3 * np.eye(n), corr_prop=0.5, seed=1
    )
    f = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
    psd = (np.abs(np.fft.rfft(x, axis=-1)) ** 2 / (sample_rate * n_samples)).mean(axis=0)

    # about 1600 bins x 32 detectors: the standard error is well below 1%
    m = (f > 0.02) & (f < 1.0)
    ratio = (psd[m] / (1 + knee / f[m])).mean()
    assert np.abs(ratio - 1) < 0.05, f"measured / model PSD = {ratio:.3f}"


def test_global_seed_reproducibility():
    """np.random.seed must reproduce the noise, and a different seed must give a different realization."""
    kwargs = dict(duration=30, sample_rate=20.0)
    (a,), _ = simulate_noise(noise_array(n=8), np_seed=1, **kwargs)
    (b,), _ = simulate_noise(noise_array(n=8), np_seed=1, **kwargs)
    (c,), _ = simulate_noise(noise_array(n=8), np_seed=2, **kwargs)

    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_seeds_give_independent_pink_noise():
    """Different global seeds must give independent 1/f noise (it came from a hard-coded key: correlation 0.98)."""
    sample_rate = 20.0
    kwargs = dict(arrays=noise_array(n=30, knee=2.0), duration=300, sample_rate=sample_rate)
    (a,), _ = simulate_noise(np_seed=1, **kwargs)
    (b,), _ = simulate_noise(np_seed=2, **kwargs)

    # the standard error is about 0.015
    corr = whitened_correlation(a, b, sample_rate, knee=2.0, f_max=1.0).mean()
    assert np.abs(corr) < 0.1, f"correlation between seeds 1 and 2: {corr:.3f}"


def test_runs_give_independent_realizations():
    """
    Each run() of a Simulation must give a new noise realization, including the 1/f part (it was the same in
    every run).
    """
    sample_rate = 20.0
    plan = get_plan("ten_second_zenith_stare", duration=300, sample_rate=sample_rate)
    instrument = maria.Instrument(arrays=[noise_array(n=30, knee=2.0)])
    sim = Simulation(instrument=instrument, plans=plan, site="llano_de_chajnantor", progress_bars=False)
    a, b = (np.asarray(sim.run(units="pW")[0].data["noise"], dtype=float) for _ in range(2))

    # the standard error is about 0.015
    corr = whitened_correlation(a, b, sample_rate, knee=2.0, f_max=1.0).mean()
    assert np.abs(corr) < 0.1, f"correlation between runs: {corr:.3f}"
    corr = whitened_correlation(a, b, sample_rate, knee=0.0).mean()
    assert np.abs(corr) < 0.1, f"white-noise correlation between runs: {corr:.3f}"


def test_observations_and_bands_are_independent():
    sample_rate = 20.0
    arrays = {
        "A": noise_array(n=30, knee=2.0),
        "B": {**noise_array(n=30, knee=2.0), "bands": {"f090": {"center": 90e9, "width": 20e9, "NET_RJ": NET_RJ}}},
    }
    noises, tods = simulate_noise(
        arrays, noise_kwargs={"correlated_noise_proportion": 0.0}, duration=300, sample_rate=sample_rate, n_plans=2
    )
    # the standard errors are about 0.01
    obs_corr = whitened_correlation(noises[0], noises[1], sample_rate, knee=2.0, f_max=1.0).mean()
    assert np.abs(obs_corr) < 0.1, f"correlation between observations: {obs_corr:.3f}"

    band_name = tods[0].dets.band_name
    f090, f150 = noises[0][band_name == "f090"], noises[0][band_name == "f150"]
    band_corr = whitened_correlation(f090, f150, sample_rate, knee=2.0, f_max=1.0).mean()
    assert np.abs(band_corr) < 0.1, f"correlation between bands: {band_corr:.3f}"


@pytest.mark.parametrize(
    "noise_kwargs",
    [
        {"correlated_noise_proportion": -0.1},
        {"correlated_noise_proportion": 1.5},
        {"correlated_noise_spatial_scale": 0.0},
        {"correlated_noise_spatial_scale": -1.0},
    ],
)
def test_out_of_range_noise_kwargs_raise(noise_kwargs):
    """A proportion outside [0, 1] gave NaN noise, and a non-positive spatial scale a degenerate basis."""
    plan = get_plan("ten_second_zenith_stare", duration=5, sample_rate=10)
    with pytest.raises(ValueError, match=list(noise_kwargs)[0]):
        Simulation(instrument="test/1deg", plans=plan, site="llano_de_chajnantor", noise_kwargs=noise_kwargs)


def test_invalid_noise_kwargs_raise():
    plan = get_plan("ten_second_zenith_stare", duration=5, sample_rate=10)
    with pytest.raises(ValueError, match="noise_kwargs"):
        Simulation(instrument="test/1deg", plans=plan, site="llano_de_chajnantor", noise_kwargs={"corr_prop": 0.1})


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_knee_is_where_pink_equals_white(beta):
    """
    The pink noise must have a two-sided PSD (knee / f) ** beta (in units of the white level), so that it equals the
    white noise at f = knee for any beta (it used to be knee / f ** beta, with the crossover at knee ** (1 / beta)).
    """
    n_samples, sample_rate, knee = 2**15, 20.0, 0.3
    x = generate_noise_with_knee(shape=(64, n_samples), sample_rate=sample_rate, knee=knee, beta=beta, seed=5)
    f = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
    psd = (np.abs(np.fft.rfft(x, axis=-1)) ** 2 / (sample_rate * n_samples)).mean(axis=0)

    m = (f > 0.05) & (f < 2.0)
    ratio = psd[m] / (1 + (knee / f[m]) ** beta)
    assert np.abs(ratio.mean() - 1) < 0.03, f"measured / model PSD = {ratio.mean():.3f}"
