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

import inspect

class AbstractRimeDataSource(object):

    def name(self):
        """ Return the name associated with this data source """
        raise NotImplementedError()

    def close(self):
        """ Perform any required cleanup """
        raise NotImplementedError()

    def clear_cache(self):
        """ Clears any caches associated with the source """
        raise NotImplementedError()

    def sources(self):
        """ Returns a dictionary of source methods, keyed on source name """
        raise NotImplementedError()

    def updated_dimensions(self):
        """ Return an iterable/mapping of hypercube dimensions to update """
        raise NotImplementedError()

    def updated_arrays(self):
        """ Return an iterable/mapping of hypercube arrays to update """
        raise NotImplementedError()

class RimeDataSource(AbstractRimeDataSource):

    SOURCE_ARGSPEC = ['self', 'context']

    def close(self):
        """ Perform any required cleanup. """
        pass

    def clear_cache(self):
        """ Clears any caches associated with the source """
        pass

    def sources(self):
        """
        Returns a dictionary of source methods found on this object,
        keyed on method name. Source methods are identified by
        (self, context) arguments on this object. For example:

        def f(self, context):
            ...

        is a source method, but

        def f(self, ctx):
            ...

        is not.

        """

        return { n: m for n, m
            in inspect.getmembers(self, inspect.ismethod)
            if inspect.getargspec(m)[0] == self.SOURCE_ARGSPEC
        }


