from __future__ import annotations

import numpy as np
from maria.weather import Weather


def test_weather_call_with_array_pressure_levels():
    weather = Weather(region="chajnantor", time="2026-01-15T04:00:00+00:00")
    from_array = weather(pressure_level=np.array([500.0, 400.0]), fields=["temperature"])
    from_list = weather(pressure_level=[500.0, 400.0], fields=["temperature"])
    assert np.allclose(from_array["temperature"].K, from_list["temperature"].K)
