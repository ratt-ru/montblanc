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
import threading
import types

import montblanc
from .source_provider import SourceProvider

def _cache(method):
    """
    Decorator for caching data source return values

    Create a key index for the proxied array in the context.
    Iterate over the array shape descriptor e.g. (ntime, nbl, 3)
    returning tuples containing the lower and upper extents
    of string dimensions. Takes (0, d) in the case of an integer
    dimensions.
    """

    @functools.wraps(method)
    def memoizer(self, context):
        # Construct the key for the given index
        idx = context.array_extents(context.name)
        key = tuple(i for t in idx for i in t)

        with self._lock:
            # Access the sub-cache for this data source
            array_cache = self._cache[context.name]

            # Cache miss, call the data source
            if key not in array_cache:
                array_cache[key] = method(context)

            return array_cache[key]

    return memoizer

def _proxy(method):
    """
    Decorator returning a method that proxies a data source.
    """
    @functools.wraps(method)
    def memoizer(self, context):
        return method(context)

    return memoizer

class CachedSourceProvider(SourceProvider):
    """
    Caches calls to data_sources on the listed providers
    """
    def __init__(self, providers, cache_data_sources=None,
                clear_start=False, clear_stop=False):
        """
        Parameters
        ----------
        providers: SourceProvider or Sequence of SourceProviders
            providers containing data sources to cache
        cache_data_sources: list of str
            list of data sources to cache (Defaults to None
            in which case all data sources are cached)
        clear_start: bool
            clear cache on start
        clear_stop: bool
            clear cache on stop
        """
        if not isinstance(providers, collections.Sequence):
            providers = [providers]

        self._cache = collections.defaultdict(dict)
        self._lock = threading.Lock()
        self._clear_start = clear_start
        self._clear_stop = clear_stop
        self._providers = providers

        # Construct a list of provider data sources
        prov_data_sources = { n: ds for prov in providers
                            for n, ds in prov.sources().iteritems() }

        # Uniquely identify data source keys
        prov_ds = set(prov_data_sources.keys())

        # Cache all data sources by default
        if cache_data_sources is None:
            cache_data_sources = prov_ds
        else:
            # Uniquely identify cached data sources
            cache_data_sources = set(cache_data_sources)
            ds_diff = list((cache_data_sources.difference(prov_ds)))

            if len(ds_diff) > 0:
                montblanc.log.warning("'{}' was requested to cache the "
                                     "following data source(s) '{}' "
                                    "but they were not present on the "
                                    "supplied providers '{}'".format(
                                        self.name(), ds_diff,
                                        [p.name() for p in providers]))


        # Construct data sources on this source provider
        for n, ds in prov_data_sources.iteritems():
            if n in cache_data_sources:
                setattr(self, n, types.MethodType(_cache(ds), self))
            else:
                setattr(self, n, types.MethodType(_proxy(ds), self))

    def init(self, init_context):
        """ Perform any initialisation required """
        for p in self._providers:
            p.init(init_context)

    def start(self, start_context):
        """ Perform any logic on solution start """
        for p in self._providers:
            p.start(start_context)

        if self._clear_start:
            self.clear_cache()

    def stop(self, stop_context):
        """ Perform any logic on solution stop """
        for p in self._providers:
            p.stop(stop_context)

        if self._clear_stop:
            self.clear_cache()

    def updated_dimensions(self):
        """ Update the dimensions """
        return [d for p in self._providers
                  for d in p.updated_dimensions()]

    def name(self):
        sub_prov_names = ', '.join([p.name() for p in self._providers])
        return 'Cache({})'.format(sub_prov_names)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def cache_size(self):
        with self._lock:
            return sum(a.nbytes for k, v in self._cache.iteritems()
                                for a in v.itervalues())