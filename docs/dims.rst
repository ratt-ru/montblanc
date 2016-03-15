Dimensions
==========

The problem space of the RIME is large. An SKA-sized problem has approximately
3000 antenna, 5,000,000 baselines and 256,000 channels. For the purposes of memory budgeting and compute sub-division. Montblanc stores dimension information in a sub-classed AttrDict object which looks something like follows:

.. code:: python

    {
        'name'        : 'ntime',
        'description' : 'Number of Timesteps',
        'global_size' : 100000,       # Global dimension size
        'local_size'  : 100,          # Local dimension size
        'extents'     : [5000, 5098], # Current problem extents
        'safety'      : True,         # Warn about local_size updates
        'zero_valid'  : True,         # Zero dimension allowed
    }

    # The following must hold
    local_size <= global_size
    extents[1] - extents[0] <= local_size

**global_size** indicates the total size of this dimension. However, not all
of this dimension may be available on the current solver object: **local_size**
indicates how much space for this dimension is available on the solver.
This figure will be used when allocating CPU or GPU arrays for the solver.

**extents** indicate the global extents of the dimension on the solver.
This implies that the current extents may not completely fill up
the **local_size**. This indicates that a smaller portion of the dimension
shoudl be considered for compute, compared to the locally available space
on the solver. **zero_valid** indicates that a zero-sized dimension is permitted.

Updates to dimension objects are permitted for a restricted subset of
attributes. **extents** may be freely updated subject to the above constraints,
while **local_size** may only be modified if **safety** is set to True.

.. code:: python

    dim.update({'extents': [5098, 6050]})