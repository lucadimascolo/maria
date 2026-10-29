from __future__ import annotations

import maria
import numpy as np
from maria.beam import compute_angular_fwhm
from maria.instrument import Band
from maria.map import ProjectionMap


def test_band_beam_uses_each_aperture(simulate_map):
    """Two apertures sharing one band each see the map smoothed with their own beam."""
    band = Band(center=150e9, width=10e9, NET_RJ=1e-5, name="f150")
    instrument = maria.Instrument(
        arrays=[
            {"n": 1, "bands": [band], "primary_size": 6.0, "name": "big"},
            {"n": 1, "bands": [band], "primary_size": 0.5, "name": "small"},
        ]
    )

    # a Gaussian source of FWHM s0 and peak A (in K_RJ), seen through a Gaussian beam of FWHM b, peaks at
    # A s0^2 / (s0^2 + b^2)
    s0, A, res = 0.05, 1e-2, 0.005
    x = res * (np.arange(601) - 300)
    X, Y = np.meshgrid(x, x)
    data = A * np.exp(-4 * np.log(2) * (X**2 + Y**2) / s0**2)
    m = ProjectionMap(data=data, resolution=res, nu=150e9, center=(150.0, -30.0), units="K_RJ")

    signal = simulate_map(m, instrument=instrument, radius=1e-4, duration=10).run(units="K_RJ")[0].signal.compute()
    for det_index, primary_size in enumerate(np.asarray(instrument.dets.primary_size)):
        b = np.degrees(compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=150e9))
        expected = A * s0**2 / (s0**2 + b**2)
        assert np.isclose(signal[det_index].mean(), expected, rtol=0.03), (primary_size, expected)


def test_map_that_already_has_a_beam(simulate_map, caplog):
    """A map that is already convolved with a beam b (its BMAJ) is only smoothed up to the band beam B."""
    primary_size, A = 10, 1e-2
    B = np.degrees(compute_angular_fwhm(fwhm_0=primary_size, z=np.inf, nu=150e9))
    res = B / 10
    x = res * (np.arange(101) - 50)
    X, Y = np.meshgrid(x, x)

    def peak(b, beam):
        # a point source seen through a beam of FWHM b, peaking at A
        data = A * np.exp(-4 * np.log(2) * (X**2 + Y**2) / b**2)
        m = ProjectionMap(data=data, resolution=res, nu=150e9, center=(150.0, -30.0), units="K_RJ", beam=beam)
        sim = simulate_map(m, primary_size=primary_size, radius=1e-4, duration=5)
        return sim.run(units="K_RJ")[0].signal.compute().mean()

    # the source is then seen through the band beam B: its peak is A b^2 / B^2 (it was A b^2 / (b^2 + B^2))
    b = 0.6 * B
    assert np.isclose(peak(b, beam=b), A * b**2 / B**2, rtol=0.01)

    # a map without a beam is smoothed with the whole band beam, as before
    assert np.isclose(peak(b, beam=0), A * b**2 / (b**2 + B**2), rtol=0.01)

    # a map with a wider beam than the band is not smoothed, with a warning
    b = 1.5 * B
    with caplog.at_level("WARNING", logger="maria"):
        assert np.isclose(peak(b, beam=b), A, rtol=0.01)
    assert "wider than the beam of band" in caplog.text
