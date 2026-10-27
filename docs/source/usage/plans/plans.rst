########
Overview
########


How an instrument observes is represented as a ``Plan``. Because it can be difficult to tell when a given source will be far enough above the horizon to observe, the easiest way to generate a plan is with a ``Planner``.

++++++++
Planners
++++++++

We can generate plans with the ``Planner`` as

.. plot:: 
   :include-source: True

    import maria
    from maria import Planner

    input_map = maria.map.load(maria.io.fetch("maps/crab_nebula.fits")) # load an example map

    planner = Planner(target=input_map, 
                      site="green_bank", 
                      constraints={"el": (70, 90), # in degrees
                                   "min_sun_distance": 20, # in degrees
                                   "hour": (14, 15)})

    plan = planner.generate_plan(total_duration=600, # in seconds
                                 scan_parameters={"radius": input_map.width.deg / 3}, # in degrees
                                 sample_rate=50) # in Hz

    plan.plot()


Passing parameters to ``generate_plan`` will be passed to the ``Plan``.

+++++++++++++++++
Customizing Plans
+++++++++++++++++

A 60-second zenith stare would be instantiated as

.. code-block:: python

    stare = maria.get_plan(start_time="2022-02-10T06:00:00",
                           scan_type="stare",
                           scan_parameters={"el_center": 0},
                           duration=60, # in seconds
                           sample_rate=20, # in Hz
                           ) # in degrees

We might also do a daisy scan on some given point on the sky, which the telescope will track:

.. code-block:: python

    tracking_daisy = maria.get_plan(start_time="2022-02-10T06:00:00",
                                    scan_type="daisy",
                                    scan_parameters={"ra_center": 150,
                                                     "dec_center": 10,
                                                     "radius": 0.5, 
                                                     "speed": 0.1,}, # in degrees
                                    duration=600, # in seconds
                                    sample_rate=50, # in Hz
                                    ) # in degrees

++++++++++++++++++++++++++++++
Scan parameters, frames, speed
++++++++++++++++++++++++++++++

The frame of a plan is inferred from the names of its scan parameters: ``az_center``/``el_center`` (and
``az_throw``/``el_throw``) give an ``az/el`` plan, ``ra_center``/``dec_center`` an ``ra/dec`` plan, and ``l_center``/``b_center``
a ``galactic`` plan. Scan parameters in different frames (e.g. ``az_center`` together with ``ra_throw``) raise a ``ValueError``.
Angles are in degrees and speeds in degrees per second (in radians with ``"degrees": False``).

The size of a scan can be given in two ways, and ``speed`` means a different thing for each:

- ``radius`` (or ``x_throw``/``y_throw``): the scan pattern is made of offsets on the sky around the center, and ``speed`` is the
  speed of the boresight on the sky.
- ``<lon>_throw`` / ``<lat>_throw`` (e.g. ``az_throw``): the pattern is added to the coordinates themselves, so ``speed`` is the
  rate of change of the coordinate. For an ``az_throw`` this is the azimuth rate of the mount, and the speed on the sky is
  ``speed * cos(el)`` (half the requested value at an elevation of 60 degrees). This is the usual convention for
  constant-elevation scans.

+++++++++++++++++++++
Pointing conventions
+++++++++++++++++++++

**Geometric az/el.** maria's azimuth and elevation are geometric (vacuum) positions: atmospheric refraction is not
modelled, and the conversion to ``ra/dec`` (ICRS) and ``galactic`` matches astropy's ``AltAz`` frame with ``pressure=0``.
Pointing positions from a real telescope or a real scan schedule are usually apparent (refracted) positions, which are
higher by about 1' at an elevation of 30 degrees, 34" at 45 degrees and 19" at 60 degrees for a high, dry site at
millimeter wavelengths. Refraction cancels in simulate-then-map round trips within maria.

**Detector offsets.** Detector offsets ``(x, y)`` around a center are a zenithal equidistant (ARC) projection: a point at
an angular distance ``r`` and position angle ``p`` from the center (measured from the direction of increasing latitude
towards increasing longitude) has ``x = -r sin(p)`` and ``y = r cos(p)``. So ``+y`` points towards increasing latitude
(elevation or declination), and ``+x`` towards *decreasing* longitude: ``x ≈ -Δaz cos(el)`` in ``az/el`` and
``x ≈ -ΔRA cos(dec)`` in ``ra/dec``. In ``ra/dec`` this is the usual sky-image orientation (north up, east to the left).
Because azimuth and right ascension increase in opposite senses on the sky, the ``az/el`` and ``ra/dec`` offset frames are
mirror images of each other (the Jacobian between them has determinant -1).

The instrument's detector offsets are applied in the ``az/el`` offset frame, after a counter-clockwise rotation (from
``+x`` towards ``+y``) by the plan's ``roll``. Seen by an observer facing the boresight, ``+x`` is to the left and ``+y``
is up. Offsets from an instrument that uses the convention ``ξ = +Δaz cos(el)`` must have their ``x`` negated.

**Polarization angles.** A detector's polarization angle ``gamma`` is applied as a fixed angle in the Q/U basis of the
sky map being sampled (the frame of that map). It does not rotate with the plan's ``roll`` or with the parallactic angle.

**Limits.** The boresight speed is checked against the instrument's azimuth speed limit (with a warning only); the
elevation speed and the acceleration are not checked.
