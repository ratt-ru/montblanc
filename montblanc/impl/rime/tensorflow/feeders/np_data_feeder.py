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

import montblanc
import montblanc.util as mbu

from rime_data_feeder import RimeDataFeeder

class NumpyRimeDataFeeder(RimeDataFeeder):
    """
    Given a dictionary containing numpy arrays and keyed on array name,
    provides feed functions for each array.


    >>> feeder = NumpyRimeDataFeeder({
        "uvw" : np.zeros(shape=(100,14,3),dtype=np.float64),
        "antenna1" : np.zeros(shape=(100,351), dtype=np.int32)})

    >>> context = FeedContext(...)
    >>> feeder.uvw(context)
    >>> feeder.antenna1(context)

    """
    def __init__(self, arrays, cube):
        self._arrays = arrays

        def _create_feed_function(name, array):
            def _feed(self, context):
                """ Generic feed function """
                idx = context.slice_index(*context.array(name).shape)
                return array[idx]

            return _feed

        # Create feed methods for each supplied array
        for n, a in arrays.iteritems():
            try:
                array_schema = cube.array(n)
            except KeyError as e:
                # Ignore the array if it isn't defined on the cube
                raise ValueError("Feed array '{n}' is not defined "
                    "on the hypercube.".format(n=n)), None, sys.exc_info()[2]

            # Except the shape of the supplied array to be equal to
            # the size of the global dimensions
            shape = tuple(cube.dim_global_size(d) if isinstance(d, str)
                else d for d in array_schema.shape)

            if shape != a.shape:
                raise ValueError("Shape of supplied array '{n}' "
                    "does not match the global shape '{g}' "
                    "of the array schema '{s}'.".format(
                        n=n, g=shape, s=array_schema.shape))

            # Create the feed function, update the wrapper,
            # bind it to a method and set the attribute on the object
            f = functools.update_wrapper(
                _create_feed_function(n, a),
                _create_feed_function)

            f.__doc__ = "Feed function for array '{n}'".format(n=n)

            method = types.MethodType(f, self)
            setattr(self, n, method)

    def feeds(self):
        return { k : getattr(self, k) for k in self.arrays }            

    @property
    def arrays(self):
        return self._arrays
