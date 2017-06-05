Concepts
--------

Montblanc predicts the model visibilities of an radio interferometer from a parametric sky model. Internally, this computation is performed via either CPUs or GPUs by Google's tensorflow_ framework.

When the number of visibilities and radio source is large, it becomes more computationally efficient to compute on GPUs. However, the problem space also becomes commensurately larger and therefore requires subdividing the problem so that *tiles*, or chunks, can fit both within the memory budget of a GPU and a CPU-only node.

HyperCubes
~~~~~~~~~~

In order to reason about tile memory requirements, Montblanc uses hypercube_ to define problem :py:class:`~hypercube.dims.Dimension`, as well as the Schemas of input, temporary result and arrays.

For example, given the following expression for computing the complex phase :math:`\phi`.

.. math::

    n &= \sqrt{1 - l^2 + m^2} - 1 \\
    \phi &= e^\frac{2\pi(ul + vm + wn)}{\lambda}

we configure a hypercube:

.. code-block:: python

    # Create cube
    from hypercube import HyperCube
    cube = HyperCube()

    # Register Dimensions
    cube.register_dimension("ntime", 10000, description="Timesteps")
    cube.register_dimension("na", 64, description="Antenna")
    cube.register_dimension("nchan", 32768, description="Channels")
    cube.register_dimension("npsrc", 100, description="Point Sources")

    # Input Array Schemas
    cube.register_arrays("lm", ("npsrc", 2), np.float64)
    cube.register_arrays("uvw", ("ntime", "na", 3), np.float64)
    cube.register_arrays("frequency", ("nchan",), np.float64)

    # Output Array Schemas
    cube.register_array("complex_phase", ("npsrc", "ntime", "na", "nchan"),
        np.complex128)

and iterate over it in tiles of 100 timesteps and 64 channels:

.. code-block:: python

    # Iterate over tiles of 100 timesteps and 64 channels
    iter_args = [("ntime", 100), ("nchan", 64)]
    for (lt, ut), (lc, uc) in cube.extent_iter(*iter_args):
        print "Time[{}:{}] Channels[{}:{}]".format(lt,ut,lc,uc)

This roduces the following output:

.. code-block:: bash

    Time[0:100] Channels[0:64]
    ...
    Time[1000:1100] Channels[1024:1088]
    ...
    Time[9900:10000] Channels[32704:32768]

Please review the hypercube  `Documentation <https://hypercube.readthedocs.io/en/latest/index.html>`_ for further information.

.. _data-sources-and-sinks:

Data Sources and Sinks
~~~~~~~~~~~~~~~~~~~~~~

The previous section illustrated how the computation of the complex phase could be subdivided. Montblanc internally uses this mechanism to perform memory budgeting and problem subdivision when computing.

Each input array, specified in the hypercube and required by Montblanc, must be supplied by the user via a *Data Source*. Conversely, output arrays are supplied to the user via a *Data Sink*. Data Sources and Sinks request and provide tiles of data and are specified on *Source* and *Sink Provider* classes:

.. code-block:: python

    lm_coords = np.ones(shape=[1000,2], np.float64)
    frequencies = np.ones(shape=[64,], np.float64)

    class MySourceProvider(SourceProvider):
        """ Data Sources """
        def lm(self, context):
            """ lm coordinate data source """
            (lp, up) = context.dim_extents("npsrc")
            return lm_coords[lp:up,:]

        def frequency(self, context):
            """ frequency data source """
            (lc, uc) = context.dim_extents("nchan")
            return frequencies[lc:uc]

        def updated_dimensions(self):
            """ Inform montblanc about global dimensions sizes """
            return [("npsrc", 1000), ("nchan", 64),
                ("ntime" , ...), ("na", ...)]

    class MySinkProvider(SinkProvider):
        """ Data Sinks """
        def complex_phase(self, context):
            """ complex phase data sink """
            (lp, up), (lt, ut), (la, ua), (lc, uc) = \
                context.dim_extents("npsrc", "ntime", "na", "nchan")

            print ("Received Complex Phase"
                   "[{}:{},{}:{},{}:{},{}:{}]"
                        .format(lp,up,lt,ut,la,ua,lc,uc))
            print "Data {}", context.data

Important points to note:

1. Data sources return a numpy data tile
   with shape :py:obj:`.SourceContext.shape`
   and dtype :py:obj:`.SourceContext.dtype`.
   :py:obj:`.SourceContext` objects have methods and attributes
   describing the *extents* of the data tile.
2. Data sinks supply a numpy data tile on the context's
   :py:obj:`.SinkContext.data` attribute.
3. :py:meth:`.AbstractSourceProvider.updated_dimensions` provides
   Montblanc with a list of dimension global sizes. This can be used
   to set the number of Point Sources, or number of Timesteps.
4. :py:meth:`.SourceContext.help`
   and :py:meth:`.SinkContext.help` return a string providing
   help describing the data sources, the extents of the data tile,
   and (optionally) the hypercube.
5. If no user-configured data source is supplied, Montblanc will
   supply default values, [0, 0] for lm coordinates and
   [1, 0, 0, 0] for stokes parameters, for example.


Provider Thread Safety
~~~~~~~~~~~~~~~~~~~~~~

**Data Sources and Sinks should be thread safe.**
Multiple calls to Data sources and sinks can be invoked from
multiple threads.
In practice, this means that if a data source is accessing
data from some `shared, mutable state <http://softwareengineering.stackexchange.com/questions/235558/what-is-state-mutable-state-and-immutable-state/235573#235573>`_, that access should be
protected by a :py:class:`threading.Lock`.

.. _hypercube: https://hypercube.readthedocs.io
.. _tensorflow: https://www.tensorflow.org
.. _numpy: https://www.numpy.org
