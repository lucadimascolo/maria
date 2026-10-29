from __future__ import annotations

import logging

import numpy as np
import pytest
from maria import Plan, get_instrument, get_site
from maria.sim.observation import Observation


def warnings_for(plan, caplog, **limits):
    instrument = get_instrument("test/1deg")
    for key, value in limits.items():
        setattr(instrument, key, value)
    with caplog.at_level(logging.WARNING, logger="maria"):
        caplog.clear()
        Observation(instrument, plan, get_site("cerro_toco"))
    return [r.getMessage() for r in caplog.records if "exceeds" in r.getMessage()]


def el_scan(speed=1.0, y_throw=2.0, sample_rate=100):
    return Plan.generate(
        scan_type="back_and_forth",
        scan_parameters={"x_throw": 0.0, "y_throw": y_throw, "speed": speed},
        frame="az/el",
        scan_center=(90.0, 50.0),
        site="cerro_toco",
        start_time=1.7e9,
        duration=30,
        sample_rate=sample_rate,
    )


def test_elevation_speed_is_checked(caplog):
    """Only the azimuth speed was checked: an elevation scan faster than the instrument's speed limit passed."""
    plan = el_scan(speed=1.0)
    assert not warnings_for(plan, caplog, az_vel_limit=2.0, acc_limit=1e3)
    messages = warnings_for(plan, caplog, az_vel_limit=0.5, acc_limit=1e3)
    assert any("elevation velocity" in m for m in messages)
    assert not any("acceleration" in m for m in messages)


@pytest.mark.parametrize("sample_rate", [100, 1000])
def test_acceleration_is_checked(caplog, sample_rate):
    """`Instrument.acc_limit` was stored but never checked. The turnarounds of a 2 deg/s elevation scan with a
    +-2 deg throw accelerate the boresight by 20.5 deg/s^2 (19-20.5 deg/s^2 over the 10 ms steps of the check); a
    stare does not warn at high sample rates."""
    plan = el_scan(speed=2.0, sample_rate=sample_rate)
    assert not warnings_for(plan, caplog, acc_limit=30.0)
    assert any("acceleration" in m for m in warnings_for(plan, caplog, acc_limit=15.0))

    stare = Plan.generate(
        scan_type="stare", frame="ra/dec", scan_center=(300.0, -30.0), site="cerro_toco", sample_rate=sample_rate,
        start_time="2024-08-06T03:00:00", duration=30,
    )  # fmt: skip
    assert not warnings_for(stare, caplog, acc_limit=1.0)


@pytest.mark.parametrize("sample_rate", [100, 1000])
def test_acceleration_of_sharp_turnarounds(caplog, sample_rate):
    """The turnarounds of a 3 deg/s elevation scan with a +-0.5 deg throw accelerate the boresight by 184 deg/s^2
    (the pattern sampled at 10 us). A gradient of the gradient over the 10 ms steps smoothed them to 143 deg/s^2,
    so a 160 deg/s^2 limit did not warn; the 3-point second difference gives 170 deg/s^2. With the 100 deg/s^2
    default, the +-0.5 deg scan warns and a +-1 deg scan (92 deg/s^2) does not."""
    plan = el_scan(speed=3.0, y_throw=0.5, sample_rate=sample_rate)
    assert any("acceleration" in m for m in warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=160.0))
    assert not warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=190.0)
    assert any("acceleration" in m for m in warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=100.0))
    assert not warnings_for(
        el_scan(speed=3.0, y_throw=1.0, sample_rate=sample_rate), caplog, az_vel_limit=10.0, acc_limit=100.0
    )


def custom_plan(t, az_deg, el_deg):
    return Plan(time=t, phi=np.radians(az_deg), theta=np.radians(el_deg), frame="az/el", site=get_site("cerro_toco"))


def test_acceleration_with_jittery_timestamps(caplog):
    """The second difference uses the actual time steps: a 1 deg, 4 s elevation sinusoid (2.47 deg/s^2) sampled at
    100 Hz with timestamps jittered by up to 30% of a step gave 13 deg/s^2 with uniform steps."""
    rng = np.random.default_rng(0)
    t = 1.7e9 + np.sort(np.arange(0, 30, 0.01) + rng.uniform(-0.3, 0.3, 3000) * 0.01)
    plan = custom_plan(t, 90.0 + 0 * t, 50.0 + np.sin(2 * np.pi * (t - t[0]) / 4))
    assert not warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=5.0)
    assert any("acceleration" in m for m in warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=2.0))


def test_azimuth_wrap_is_not_a_jump(caplog):
    """An azimuth scan at 1 deg/s through 0/360 deg must not see the wrap as a 360 deg step (18000 deg/s and
    3.6e6 deg/s^2 at 100 Hz)."""
    t = 1.7e9 + np.arange(0, 30, 0.01)
    plan = custom_plan(t, np.mod(354.0 + (t - t[0]), 360.0), 50.0 + 0 * t)
    assert not warnings_for(plan, caplog, az_vel_limit=2.0, acc_limit=1.0)
    assert any("maximum velocity" in m for m in warnings_for(plan, caplog, az_vel_limit=0.5, acc_limit=1.0))


@pytest.mark.parametrize("defect", ["repeated timestamp", "NaN elevation"])
def test_motion_checks_survive_bad_samples(caplog, defect):
    """A repeated timestamp (a zero time step) or a NaN in the pointing gave a NaN acceleration, and max(0, nan) is
    0, so the acceleration check (and, for NaN, the elevation speed check) silently passed. The 1 deg, 4 s
    elevation sinusoid reaches 2.47 deg/s^2 and 1.57 deg/s, which must still be reported."""
    t = 1.7e9 + np.arange(0, 30, 0.01)
    el = 50.0 + np.sin(2 * np.pi * (t - t[0]) / 4)
    if defect == "repeated timestamp":
        t[500] = t[499]
    else:
        el[100] = np.nan
    plan = custom_plan(t, 90.0 + 0 * t, el)
    assert not any("acceleration" in m for m in warnings_for(plan, caplog, az_vel_limit=10.0, acc_limit=5.0))
    messages = warnings_for(plan, caplog, az_vel_limit=1.0, acc_limit=2.0)
    assert any("acceleration" in m for m in messages)
    assert any("elevation velocity" in m for m in messages)
