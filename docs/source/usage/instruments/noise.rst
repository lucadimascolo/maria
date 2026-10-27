.. _noise:

#####
Noise
#####

We can control the white noise level of the detectors in a band with either an explicit noise-equivalent power level (in Watts) as

.. code-block:: python

    from maria import Band

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   NEP=1e-15, # in W√s
                   efficiency=0.5)

or with an implicit Rayleigh-Jeans sensitivity as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5,
                   NET_RJ=1e-5) # in K_RJ√s

or in CMB units as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5,
                   NET_CMB=1e-5) # in K_CMB√s

Since the sensitivity on the sky in temperature units depends on both the instrument and the observing conditions (i.e. the atmosphere), this method computes an NEP assuming there is no atmosphere. To include the effects of atmospheric opacity in this conversion, we can supply some ``spectrum_kwargs`` as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5,
                   NET_RJ=1e-5, # in K_RJ√s
                   spectrum_kwargs={"region": "chajnantor", 
                                    "zenith_pwv": 1e0, # in mm
                                    "elevation": 45}) # in degrees


An NET is the white noise level of each detector's time-ordered data. For a polarized detector, the data in
temperature units are referred to Stokes I (a uniform map of 1 K_RJ reads 1 K_RJ), and the detector absorbs half of
the power of an unpolarized source, so it needs half the NEP of an unpolarized detector for the same NET. An NEP is
the same for every detector, so a polarized detector in a band specified by its NEP has twice the NET of an
unpolarized one. The ``NEP``, ``NET_RJ`` and ``NET_CMB`` properties of a band are for unpolarized detectors. The
NET is the noise level without an atmosphere: data in ``K_RJ`` or ``K_CMB`` are corrected for the atmospheric
extinction, so with an atmosphere the measured white noise level is the NET divided by the band's atmospheric
transmission.

+++++++++++++++
Non-white noise
+++++++++++++++

Bolometers are typically dominated by non-white noise below a certain frequency threshold (the "knee"). We can add in pink noise to our band by supplying a ``knee`` parameter (in Hz).


+++++++++++++++++++
Correlated noise
+++++++++++++++++++

A fraction of the pink noise is correlated between the detectors of a band. It is controlled by the
``noise_kwargs`` of the ``Simulation``:

.. code-block:: python

    sim = Simulation(...,
                     noise_kwargs={"correlated_noise_proportion": 0.5, # fraction of the pink noise power in correlated modes
                                   "correlated_noise_spatial_scale": 1.0}) # correlation length, as a fraction of the band's field of view

The correlated modes contain only pink noise: the white noise of every detector is independent. Bands with
more than 16 detectors get five spatially smooth modes, and smaller bands get a single common mode.

The noise (white, pink and correlated) is drawn from NumPy's global random state, so ``np.random.seed`` makes a
simulation reproducible, and each call of ``run()`` gives a new realization.

++++++++++++++++++++++++++
Loading-dependent noise
++++++++++++++++++++++++++

A band's ``NEP_per_loading`` (default 0) adds noise that grows with the optical loading of each detector: the NEP
used for each sample is ``NEP + NEP_per_loading * P``, with ``NEP_per_loading`` in W√s per pW of loading and ``P``
the total simulated loading in pW (atmosphere and sky), clipped at zero. This is a linear model, not a
photon-noise (√P) model.

Detector time constants are not implemented: a band's ``time_constant`` is stored but not applied to the
simulated data, and a non-zero value logs a warning.
