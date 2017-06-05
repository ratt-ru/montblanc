import itertools

import numpy as np

# These are hypercube dimension attributes
DEFAULT_SCHEMA = ['lower_extent', 'upper_extent', 'global_size']

class CubeDimensionTranscoder(object):
    """
    Small class for encoding and decoding hypercube dimensions
    into/from numpy arrays

    >>> import hypercube as hc
    >>> cube = hc.HyperCube()
    >>> cube.register_dimension('ntime', 100)
    >>> cube.register_dimension('na', 7)
    >>> cube.register_dimension('nchan', 128)

    >>> dimdesc = CubeDimensionTranscoder(['ntime', 'nchan'])
    >>> desc = dimdesc.encode(cube.dimensions())
    >>> print desc
    >>> time, chan = dimdesc.decode(desc)
    >>> print time, chan
    """
    def __init__(self, dimensions, schema=None):
        if schema is None:
            schema = DEFAULT_SCHEMA
        elif not all([s in DEFAULT_SCHEMA for s in schema]):
            raise ValueError("Schema '{s}' contains invalid attributes. "
                "Valid attributes are '{v}'".format(s=schema, v=DEFAULT_SCHEMA))

        self._dimensions = dimensions
        self._schema = tuple(schema)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def schema(self):
        return self._schema

    def encode(self, cube_dimensions):
        """
        Produces a numpy array of integers which encode
        the supplied cube dimensions.
        """
        return np.asarray([getattr(cube_dimensions[d], s)
            for d in self._dimensions
            for s in self._schema],
                dtype=np.int32)

    def decode(self, descriptor):
        """ Produce a list of dictionaries for each dimension in this transcoder """
        i = iter(descriptor)
        n = len(self._schema)

        # Add the name key to our schema
        schema = self._schema + ('name',)
        # For each dimensions, generator takes n items off iterator
        # wrapping the descriptor, making a tuple with the dimension
        # name appended
        tuple_gen = (tuple(itertools.islice(i, n)) + (d, )
            for d in self._dimensions)

        # Generate dictionary by mapping schema keys to generated tuples
        return [{ k: v for k, v in zip(schema, t) } for t in tuple_gen]