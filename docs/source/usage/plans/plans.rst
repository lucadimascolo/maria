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
