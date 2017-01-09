#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import functools
import sys
import types
import unittest

import montblanc
import montblanc.util as mbu

from source_provider import SourceProvider

class NumpySourceProvider(SourceProvider):
    """
    Given a dictionary containing numpy arrays and keyed on array name,
    provides source functions for each array.


    >>> source = NumpySourceProvider({
            "uvw" : np.zeros(shape=(100,14,3),dtype=np.float64),
            "antenna1" : np.zeros(shape=(100,351), dtype=np.int32)
        }, cube)

    >>> context = SourceContext(...)
    >>> source.uvw(context)
    >>> source.antenna1(context)

    """
    def __init__(self, arrays):
        self._arrays = arrays

        def _create_source_function(name, array):
            def _source(self, context):
                """ Generic source function """
                return array[context.array_slice_index(name)]

            return _source

        # Create source methods for each supplied array
        for n, a in arrays.iteritems():
            # Create the source function, update the wrapper,
            # bind it to a method and set the attribute on the object
            f = functools.update_wrapper(
                _create_source_function(n, a),
                _create_source_function)

            f.__doc__ = "Feed function for array '{n}'".format(n=n)

            method = types.MethodType(f, self)
            setattr(self, n, method)

    @property
    def arrays(self):
        return self._arrays

    def __str__(self):
        return self.__class__.__name__

class TestNumpySourceProvider(unittest.TestCase):
    def test_numpy_source_provider(self):
        import hypercube
        import numpy as np

        # Hypercube with ntime, na dimensions
        # and a uvw array
        cube = hypercube.HyperCube()
        cube.register_dimension('ntime', 100)
        cube.register_dimension('na', 64)

        cube.register_array('uvw', ('ntime', 'na', 3), np.float64)

        # Set time and antenna extents
        lt, ut = 10, 50
        la, ua = 10, 20

        # Update dimension extents
        cube.update_dimension('ntime', lower_extent=lt, upper_extent=ut)
        cube.update_dimension('na', lower_extent=la, upper_extent=ua)

        uvw_schema = cube.array('uvw')
        global_uvw_shape = cube.dim_global_size(*uvw_schema.shape)
        uvw = (np.arange(np.product(global_uvw_shape))
                    .reshape(global_uvw_shape)
                    .astype(np.float64))

        # Create a Numpy Source Provider
        source_prov = NumpySourceProvider({"uvw" : uvw})

        class Context(object):
            """ Mock up a context object """
            def __init__(self, array, cube):
                self._cube = cube
                self.array_slice_index = cube.array_slice_index

                array_schema = cube.array(array)

                self.shape = cube.dim_extent_size(*array_schema.shape)
                self.dtype = array_schema.dtype


        data = source_prov.uvw(Context('uvw', cube))
        uvw_slice = uvw[lt:ut, la:ua, :]

        # Check that we've got the shape defined by
        # cube extents and the given dtype
        self.assertTrue(np.all(data == uvw_slice))
        self.assertTrue(data.shape == uvw_slice.shape)
        self.assertTrue(data.dtype == uvw_slice.dtype)

if __name__ == "__main__":
    unittest.main()
