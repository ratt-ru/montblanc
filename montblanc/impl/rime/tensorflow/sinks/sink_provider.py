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

class AbstractSinkProvider(object):

    def name(self):
        """ Returns this data sink's name """
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

    def clear_cache(self):
        """ Clears any caches associated with the sink """
        raise NotImplementedError()

    def sinks(self):
        """ Returns a dictionary of sink methods, keyed on sink name """
        raise NotImplementedError()

def find_sinks(obj):
    """
    Returns a dictionary of sink methods found on this object,
    keyed on method name. Sink methods are identified by
    (self, context) arguments on this object. For example:

    def f(self, context):
        ...

    is a sink method, but

    def f(self, ctx):
        ...

    is not.

    """
    SINK_ARGSPEC = ['self', 'context']

    return { n: m for n, m in inspect.getmembers(obj, inspect.ismethod)
        if inspect.getargspec(m)[0] == SINK_ARGSPEC }

class SinkProvider(AbstractSinkProvider):

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
        """ Perform any required cleanup. NOOP """
        pass

    def clear_cache(self):
        """ Clears any caches associated with the sink """
        pass

    def sinks(self):
        """
        Returns a dictionary of sink methods found on this object,
        keyed on method name. Sink methods are identified by
        (self, context) arguments on this object. For example:

        def f(self, context):
            ...

        is a sink method, but

        def f(self, ctx):
            ...

        is not.

        """

        try:
            return self._sinks
        except AttributeError:
            self._sinks = find_sinks(self)

        return self._sinks

    def __str__(self):
        return self.name()

