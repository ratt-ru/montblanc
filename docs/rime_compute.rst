Computing the RIME
------------------

Montblanc solves the Radio Inteferometer Measurement Equation (RIME).

Example Source Providers
~~~~~~~~~~~~~~~~~~~~~~~~

Although it is possible to provide custom :ref:`Source Providers <data-sources-and-sinks>` for Montblanc's inputs,
the common use case is to specify parameterised Radio Sources.

Here is a Source Provider that supplies Point Sources to Montblanc
in the form of three numpy arrays containing the
lm coordinates, stokes parameters and spectral indices,
respectively.

.. code-block:: python

    class PointSourceProvider(SourceProvider):
        def __init__(self, pt_lm, pt_stokes, pt_alpha):
            # Store some numpy arrays
            self._pt_lm = pt_lm
            self._pt_stokes = pt_stokes
            self._pt_alpha = pt_alpha

        def name(self):
            return "PointSourceProvider"

        def point_lm(self, context):
            """ Point lm data source """
            lp, up = context.dim_extents('npsrc')
            return self._pt_lm[lp:up, :]

        def point_stokes(self, context):
            """ Point stokes data source """
            (lp, up), (lt, ut) = context.dim_extents('npsrc', 'ntime')
            return np.tile(self._pt_stokes[lp:up, np.newaxis, :],
                [1, ut-lt, 1])

        def point_alpha(self, context):
            """ Point alpha data source """
            (lp, up), (lt, ut) = context.dim_extents('npsrc', 'ntime')
            return np.tile(self._pt_alpha[lp:up, np.newaxis],
                [1, ut-lt])

        def updated_dimensions(self):
            """
            Inform montblanc about the number of
            point sources to process
            """
            return [('npsrc', self._pt_lm.shape[0])]

Similarly, here is a Source Provider that supplies Gaussian Sources
to Montblanc in four numpy arrays containing the
lm coordinates, stokes parameters,  spectral indices and
gaussian shape parameters respectively.

.. code-block:: python

    class GaussianSourceProvider(SourceProvider):
        def __init__(self, g_lm, g_stokes, g_alpha, g_shape):
            # Store some numpy arrays
            self._g_lm = g_lm
            self._g_stokes = g_stokes
            self._g_alpha = g_alpha
            self._g_shape = g_shape

        def name(self):
            return "GaussianSourceProvider"

        def gaussian_lm(self, context):
            """ Gaussian lm coordinate data source """
            lg, ug = context.dim_extents('ngsrc')
            return self._g_lm[lg:ug, :]

        def gaussian_stokes(self, context):
            """ Gaussian stokes data source """
            (lg, ug), (lt, ut) = context.dim_extents('ngsrc', 'ntime')
            return np.tile(self._g_stokes[lg:ug, np.newaxis, :],
                [1, ut-lt, 1])

        def gaussian_alpha(self, context):
            """ Gaussian alpha data source """
            (lg, ug), (lt, ut) = context.dim_extents('ngsrc', 'ntime')
            return np.tile(self._g_alpha[lg:ug, np.newaxis],
                [1, ut-lt])

        def gaussian_shape(self, context):
            """ Gaussian shape data source """
            (lg, ug) = context.dim_extents('ngsrc')
            gauss_shape = self._g_shape[:,lg:ug]
            emaj = gauss_shape[0]
            emin = gauss_shape[1]
            pa = gauss_shape[2]

            gauss = np.empty(context.shape, dtype=context.dtype)

            # Convert from (emaj, emin, position angle)
            # to (lproj, mproj, ratio)
            gauss[0,:] = emaj * np.sin(pa)
            gauss[1,:] = emaj * np.cos(pa)
            emaj[emaj == 0.0] = 1.0
            gauss[2,:] = emin / emaj

            return gauss

        def updated_dimensions(self):
            """
            Inform montblanc about the number of
            gaussian sources to process
            """
            return [ ('ngsrc', self._g_lm.shape[0])]

These Source Providers are passed to the solver when
computing the RIME.


Configuring and Executing a Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly we configure the solver. Presently, this
is simple:

.. code-block:: python

    import montblanc

    slvr_cfg = montblanc.rime_solver_cfg(dtype='double',
        version='tf', mem_budget=4*1024*1024*1024)

`dtype` is either `float` or `double` and defines whether single
or double floating point precision should be used to perform computation.

Next, the RIME solver should be created, using the configuration.

.. code-block:: python

    with montblanc.rime_solver(slvr_cfg) as slvr:

Then, source and sink providers can be configured in lists
and supplied to the `solve` method on the solver:

.. code-block:: python

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Create a MS manager object, used by
        # MSSourceProvider and MSSinkProvider
        ms_mgr = MeasurementSetManager('WSRT.MS', slvr_cfg)

        source_provs = []
        source_provs.append(MSSourceProvider(ms_mgr, cache=True))
        source_provs.append(FitsBeamSourceProvider(
            "beam_$(corr)_$(reim).fits", cache=True))
        source_provs.append(PointSourceProvider)
        source_provs.append(GaussianSourceProvider)

        sink_provs = [MSSinkProvider(ms_mgr, 'MODEL_DATA')]

        slvr.solve(source_providers=source_provs,
            sink_providers=sink_provs)