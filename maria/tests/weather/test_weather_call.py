from __future__ import annotations

import numpy as np
import pytest
from maria.units import Quantity
from maria.weather import Weather


def test_weather_call_with_array_pressure_levels():
    weather = Weather(region="chajnantor", time="2026-01-15T04:00:00+00:00")
    from_array = weather(pressure_level=np.array([500.0, 400.0]), fields=["temperature"])
    from_list = weather(pressure_level=[500.0, 400.0], fields=["temperature"])
    assert np.allclose(from_array["temperature"].K, from_list["temperature"].K)


def test_wind_bearing_is_the_direction_the_wind_blows_from():
    weather = Weather(region="chajnantor", time="2026-01-15T04:00:00+00:00")
    levels = weather.data["levels"]
    n = len(levels["wind_east"])
    for wind_east, wind_north, bearing_deg in [(1.0, 0.0, 270.0), (0.0, 1.0, 180.0), (-1.0, 0.0, 90.0), (0.0, -1.0, 0.0)]:
        levels["wind_east"] = Quantity(np.full(n, wind_east), "m/s")
        levels["wind_north"] = Quantity(np.full(n, wind_north), "m/s")
        assert np.allclose(np.degrees(weather.wind_bearing) % 360, bearing_deg)


def test_unknown_weather_quantiles_raise():
    """A quantile for a field the weather does not have (e.g. "pwv") used to be ignored silently."""
    with pytest.raises(ValueError, match="pwv"):
        Weather(region="chajnantor", time="2026-01-15T04:00:00+00:00", quantiles={"pwv": 0.9})
