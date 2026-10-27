from __future__ import annotations

import copy

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from maria import Planner
from maria.coords import Frame
from maria.instrument import get_instrument
from maria.plan import PLAN_CONFIGS, Plan, get_plan
from maria.plan.patterns import back_and_forth, double_circle, lissajous, parse_scan_kwargs
from maria.plan.plan import parse_scan_parameters
from maria.site import get_site
from maria.units import Quantity

T0 = 1.7e9


def stare(start_time=T0, duration=10, sample_rate=10, **kwargs):
    return Plan.generate(
        start_time=start_time,
        duration=duration,
        sample_rate=sample_rate,
        scan_type="stare",
        frame="az/el",
        scan_center=(10, 50),
        **kwargs,
    )


@pytest.mark.parametrize("x_throw, y_throw", [(2.0, 0.0), (1.0, 0.5)])
def test_back_and_forth_throw_and_speed(x_throw, y_throw):
    """P6: back_and_forth scanned 1.099x wider and faster than requested."""
    t = T0 + np.arange(0, 120, 1e-3)
    speed = 0.5
    x, y = back_and_forth(t, x_throw=x_throw, y_throw=y_throw, speed=speed).T

    v = np.hypot(np.gradient(x, t), np.gradient(y, t))

    assert np.isclose(np.abs(x).max(), x_throw, rtol=1e-3)
    assert np.isclose(np.abs(y).max(), y_throw, rtol=1e-3, atol=1e-12)
    assert np.isclose(v.max(), speed, rtol=1e-2)


def test_back_and_forth_max_accel():
    """P6: the acceleration limit was also exceeded by 1.099x."""
    t = T0 + np.arange(0, 120, 1e-3)
    max_accel = 0.2
    x, y = back_and_forth(t, x_throw=2.0, speed=10.0, max_accel=max_accel).T

    a = np.abs(np.gradient(np.gradient(x, t), t))

    assert np.isclose(np.abs(x).max(), 2.0, rtol=1e-3)
    assert np.isclose(a.max(), max_accel, rtol=1e-2)


@pytest.mark.parametrize("pattern", [back_and_forth, lissajous, double_circle])
def test_pattern_does_not_depend_on_start_time(pattern):
    """P6: the phase was computed from the absolute unix time, so the start of the scan depended on the start time."""
    t = np.arange(0, 60, 1e-2)
    offsets = [pattern(t0 + t, x_throw=2.0, y_throw=1.0, speed=0.5) for t0 in [T0, T0 + 3.0, T0 + 1234.5]]

    for o in offsets[1:]:
        assert np.allclose(o, offsets[0], atol=1e-9)


def test_generate_does_not_mutate_scan_parameters():
    """P7: Plan.generate wrote the scan centre into its mutable default and into the caller's dict."""
    scan_parameters = {"radius": 1.0}
    Plan.generate(start_time=T0, duration=10, scan_parameters=scan_parameters, frame="ra/dec", scan_center=(150.0, -20.0))
    assert scan_parameters == {"radius": 1.0}

    # without a centre, the next plan must not inherit (150, -20) from the default
    plan = Plan.generate(start_time=T0, duration=10, scan_type="stare")
    assert plan.generation_kwargs["scan_parameters"] == {}


def test_get_plan_does_not_mutate_the_plan_library():
    """P-new1: get_plan_config made a shallow copy, so Plan.generate overwrote the built-in plan library."""
    library = copy.deepcopy(PLAN_CONFIGS)
    get_plan("ten_second_zenith_stare", frame="ra/dec", scan_center=(150.0, -20.0), start_time=T0)
    assert PLAN_CONFIGS == library

    plan = get_plan("ten_second_zenith_stare", start_time=T0)
    assert plan.frame.name == "az/el"
    assert np.isclose(np.degrees(np.float64(plan.center()[1].rad)), 90.0, atol=1e-4)


def test_parse_scan_kwargs_does_not_mutate_its_input():
    """P7 (same pattern): parse_scan_kwargs mutated its input, e.g. the default of Planner.generate_plans."""
    scan_kwargs = {}
    parse_scan_kwargs(scan_kwargs, default_radius=2.0)
    assert scan_kwargs == {}


def test_generate_with_instrument_and_no_sample_rate():
    """P8: Plan.generate(instrument=..., sample_rate=None) crashed in np.arange."""
    plan = stare(instrument="test/1deg", sample_rate=None)
    assert np.isclose(plan.sample_rate.Hz, 100.0)
    plan = stare(instrument=get_instrument("test/1deg"), sample_rate=None)
    assert np.isclose(plan.sample_rate.Hz, 100.0)
    assert plan.n == 1000
    # as on master: without a sample rate, a name is looked up and other types raise
    with pytest.raises(TypeError):
        stare(instrument=1.0, sample_rate=None)
    with pytest.raises(KeyError):
        stare(instrument="not-an-instrument", sample_rate=None)
    # as on master: with a sample rate, the instrument is not used
    assert stare(instrument=1.0, sample_rate=10).n == 100
