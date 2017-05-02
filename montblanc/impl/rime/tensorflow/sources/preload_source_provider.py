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
import itertools
import types

from .source_provider import SourceProvider
from .source_context import SourceContext

class PreloadSourceProvider(SourceProvider):
    """ Preloads given data sources from a given list of providers """

    def __init__(self, providers, preload=None):
        """
        Parameters
        ----------
        providers: SourceProvider or list of providers
            providers containing data sources to preload
        preload: list of str
            list of data source to preload (Defaults to None
            in which case all data sources are preloaded)
        """
        if not isinstance(providers, collections.Sequence):
            providers = [providers]

        self._providers = providers

        # Construct a list of data provider sources
        data_sources = { n: ds for prov in providers
                for n, ds in prov.sources().iteritems() }

        # Preload all data sources by default
        if preload is None:
            preload = data_sources.keys()

        # Graft the appropriate data sources onto this
        # Provider, applying a cache decorator
        for n in preload:
            setattr(self, n, data_sources[n])

        sources = ", ".join(self.sources().keys())
        prov_names = ", ".join([p.name() for p in self._providers])
        self._name = ''.join(("Preload([", prov_names, "], [", sources, "])"))

    def name(self):
        return self._name

    def updated_dimensions(self):
        """ Update the dimensions """
        return [d for p in self._providers
                  for d in p.updated_dimensions()]

    def start(self, start_context):
        # Preload all data sources when solving starts
        cube = start_context.cube
        array_schemas = cube.arrays(reify=True)

        it = cube.cube_iter(*start_context.iter_args, arrays=True)

        for iter_cube in it:
            for n, data_source in self.sources().iteritems():
                ctx = SourceContext(n, iter_cube,
                        start_context.cfg, start_context.iter_args,
                        cube.array(n) if n in cube.arrays() else {},
                        array_schemas[n].shape,
                        array_schemas[n].dtype)

                # Invoke the data source
                data_source(ctx)
