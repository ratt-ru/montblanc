import itertools

import numpy as np

DEFAULT_SCHEMA = ['lower_extent', 'upper_extent', 'local_size', 'global_size']

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
        self._dimensions = dimensions

        if schema is None:
            schema = DEFAULT_SCHEMA
        elif not all([s in DEFAULT_SCHEMA for s in schema]):
            raise ValueError("Schema '{s}' contains invalid attributes. "
                "Valid attributes are '{v}'".format(s=schema, v=DEFAULT_SCHEMA))

        self._schema = schema

    def encode(self, cube_dimensions):
        """
        Produces a numpy array of integers which transcode the supplied cube dimensions
        """
        return np.asarray([getattr(cube_dimensions[d], s)
            for d in self._dimensions
            for s in self._schema],
                dtype=np.int32)

    def decode(self, descriptor):
        """ Yield tuples for each dimension in this transcoder """
        i = iter(descriptor)
        n = len(self._schema)
        piece = tuple(itertools.islice(i, n))
        while piece:
            yield piece
            piece = tuple(itertools.islice(i, n))

