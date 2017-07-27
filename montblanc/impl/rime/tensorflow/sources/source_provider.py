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

class AbstractSourceProvider(object):

    def name(self):
        """ Return the name associated with this data source """
        raise NotImplementedError()

    def init(self, init_context):
        """ Called when initialising Providers """
        raise NotImplementedError()

    def start(self, start_context):
        """ Called at the start of any solution """
        raise NotImplementedError()

    def stop(self, stop_context):
        """ Called at the end of any solution """
        raise NotImplementedError()

    def close(self):
        """ Perform any required cleanup """
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

DEFAULT_ARGSPEC = ['self', 'context']

def find_sources(obj, argspec=None):
    """
    Returns a dictionary of source methods found on this object,
    keyed on method name. Source methods are identified.by argspec,
    a list of argument specifiers. So for e.g. an argpsec of
    :code:`[['self', 'context'], ['s', 'c']]` would match
    methods looking like:

    .. code-block:: python

        def f(self, context):
        ...

    .. code-block:: python

        def f(s, c):
        ...

    is but not

    .. code-block:: python

        def f(self, ctx):
        ...


    """

    if argspec is None:
        argspec = [DEFAULT_ARGSPEC]

    return { n: m for n, m in inspect.getmembers(obj, callable)
        if not n.startswith('_') and
        inspect.getargspec(m).args in argspec }


class SourceProvider(AbstractSourceProvider):

    def init(self, init_context):
        """ Called when initialising Providers """
        pass

    def start(self, start_context):
        """ Called at the start of any solution """
        pass

    def stop(self, stop_context):
        """ Called at the end of any solution """
        pass

    def close(self):
        """ Perform any required cleanup. """
        pass

    def sources(self):
        """
        Returns a dictionary of source methods found on this object,
        keyed on method name. Source methods are identified by
        (self, context) arguments on this object. For example:

        .. code-block:: python

            def f(self, context):
                    ...

        is a source method, but

        .. code-block:: python

            def f(self, ctx):
                ...

            is not.

        """

        try:
            return self._sources
        except AttributeError:
            self._sources = find_sources(self)

        return self._sources

    def updated_dimensions(self):
        """ Return an iterable/mapping of hypercube dimensions to update """
        return ()

    def updated_arrays(self):
        """ Return an iterable/mapping of hypercube arrays to update """
        return ()

    def __str__(self):
        return self.name()

