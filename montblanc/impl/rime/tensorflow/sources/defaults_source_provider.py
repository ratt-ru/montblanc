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

import collections
import functools
import unittest

from montblanc.impl.rime.tensorflow.sources.source_provider import (
    SourceProvider,
    find_sources,
    DEFAULT_ARGSPEC)

def constant_cache(method):
    """
    Caches constant arrays associated with an array name.

    The intent of this decorator is to avoid the cost
    of recreating and storing many arrays of constant data,
    especially data created by np.zeros or np.ones.
    Instead, a single array of the first given shape is created
    and any further requests for constant data of the same
    (or smaller) shape are served from the cache.

    Requests for larger shapes or different types are regarded
    as a cache miss and will result in replacement of the
    existing cache value.
    """
    @functools.wraps(method)
    def wrapper(self, context):
        # Defer to method if no caching is enabled
        if not self._is_cached:
            return method(self, context)

        name = context.name
        cached = self._constant_cache.get(name, None)

        # No cached value, call method and return
        if cached is None:
            data = self._constant_cache[name] = method(self, context)
            return data

        # Can we just slice the existing cache entry?
        # 1. Are all context.shape's entries less than or equal
        #    to the shape of the cached data?
        # 2. Do they have the same dtype?
        cached_ok = (cached.dtype == context.dtype and
            all(l <= r for l,r in zip(context.shape, cached.shape)))

        # Need to return something bigger or a different type
        if not cached_ok:
            data = self._constant_cache[name] = method(self, context)
            return data

        # Otherwise slice the cached data
        return cached[tuple(slice(0, s) for s in context.shape)]

    f = wrapper
    f.__decorator__ = constant_cache.__name__

    return f

def chunk_cache(method):
    """
    Caches chunks of default data.

    This decorator caches generated default data so as to
    avoid recomputing it on a subsequent queries to the
    provider.
    """

    @functools.wraps(method)
    def wrapper(self, context):
        # Defer to the method if no caching is enabled
        if not self._is_cached:
            return method(self, context)

        # Construct the key for the given index
        name = context.name
        idx = context.array_extents(name)
        key = tuple(i for t in idx for i in t)
        # Access the sub-cache for this array
        array_cache = self._chunk_cache[name]

        # Cache miss, call the function
        if key not in array_cache:
            array_cache[key] = method(self, context)

        return array_cache[key]

    f = wrapper
    f.__decorator__ = chunk_cache.__name__
    return f

class DefaultsSourceProvider(SourceProvider):
    def __init__(self, cache=False):
        self._is_cached = cache
        self._constant_cache = {}
        self._chunk_cache = collections.defaultdict(dict)

    def name(self):
        return self.__class__.__name__

    def clear_cache(self):
        self._constant_cache.clear()
        self._chunk_cache.clear()

class TestDefaultsSourceProvider(unittest.TestCase):

    def test_defaults_source_provider(self):
        import numpy as np
        import types

        # Create source provider and graft a model_vis method onto it
        defprov = DefaultsSourceProvider(cache=True)

        model_vis = lambda self, context: np.zeros(context.shape,
            context.dtype)

        defprov.model_vis = types.MethodType(constant_cache(model_vis),
            defprov)

        # Mock a context context object
        class Context(object):
            pass

        context = Context()
        context.name = 'model_vis'
        context.shape = (10, 16)
        context.dtype = np.float64

        A = defprov.model_vis(context)
        self.assertTrue(A.flags['OWNDATA'])
        self.assertEqual(A.shape, context.shape)

        context.name = 'model_vis'
        context.shape = supplied_shape = (100, 32)
        context.dtype = np.float64

        B = defprov.model_vis(context)
        self.assertTrue(B.flags['OWNDATA'])
        self.assertEqual(B.shape, context.shape)

        context.name = 'model_vis'
        context.shape = (8, 2)
        context.dtype = np.float64

        C = defprov.model_vis(context)
        self.assertFalse(C.flags['OWNDATA'])
        self.assertEqual(C.shape, context.shape)
        self.assertIs(C.base, B)

        cached_shape = defprov._constant_cache['model_vis'].shape
        self.assertEqual(cached_shape, supplied_shape)

if __name__ == "__main__":
    unittest.main()

