API
===

Contexts
~~~~~~~~

Contexts are objects supplying information to implementers of Providers.

.. module:: montblanc.impl.rime.tensorflow.init_context

.. autoclass:: InitialisationContext()
    :members: cfg

.. module:: montblanc.impl.rime.tensorflow.start_context

.. autoclass:: StartContext()
    :members: cube, cfg

.. module:: montblanc.impl.rime.tensorflow.stop_context

.. autoclass:: StopContext()
    :members: cube, cfg

.. module:: montblanc.impl.rime.tensorflow.sources.source_context

.. autoclass:: SourceContext()
    :members: cube, cfg, shape, dtype, name,
                array_schema, iter_args, help,
                array_extents,
                dim_global_size, dim_extents,
                dim_lower_extent, dim_upper_extent

.. module:: montblanc.impl.rime.tensorflow.sinks.sink_context

.. autoclass:: SinkContext()
    :members: cube, cfg, data, name, array_schema, iter_args, help,
                input, array_extents,
                dim_global_size, dim_extents,
                dim_lower_extent, dim_upper_extent



Abstract Provider Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

This is the Abstract Base Class that all Source Providers must inherit from.
Alternatively, the :py:class:`~montblanc.impl.rime.tensorflow.sources.source_provider.SourceProvider` class inherits from AbstractSourceProvider
and provides some useful concrete implementations.


.. module:: montblanc.impl.rime.tensorflow.sources.source_provider

.. autoclass:: AbstractSourceProvider
    :members:

This is the Abstract Base Class that all Sink Providers must inherit from.
Alternatively, the :py:class:`~montblanc.impl.rime.tensorflow.sinks.sink_provider.SinkProvider` class inherits from AbstractSinkProvider
and provides some useful concrete implementations.

.. module:: montblanc.impl.rime.tensorflow.sinks.sink_provider

.. autoclass:: AbstractSinkProvider
    :members:

Source Provider Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: montblanc.impl.rime.tensorflow.sources.ms_source_provider

.. autoclass:: MSSourceProvider()
    :members:

    .. automethod:: __init__


.. module:: montblanc.impl.rime.tensorflow.sources.fits_beam_source_provider

.. autoclass:: FitsBeamSourceProvider()
    :members:

    .. automethod:: __init__

Sink Provider Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: montblanc.impl.rime.tensorflow.sinks.ms_sink_provider

.. autoclass:: MSSinkProvider()
    :members:

    .. automethod:: __init__