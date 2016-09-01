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

def _get_public_methods(obj):
    """ Return the public methods on an object """
    return set(n for n, m
        in inspect.getmembers(obj, inspect.ismethod)
        if not n.startswith('_'))

class _setter_property(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)

class SinkContext(object):
    """
    Context for queue arrays.

    Proxies methods of a hypercube and provides access to configuration
    """
    __slots__ = ('_cube', '_cfg', '_name', '_array',
        '_cube_methods')

    def __init__(self, name, cube, slvr_cfg, array):
        self._name = name
        self._cube = cube
        self._cfg = slvr_cfg
        self._array = array
        self._cube_methods = _get_public_methods(cube)

        # Fall over if there's any intersection between the
        # public methods on the hypercube, the current class
        # and the cfg
        intersect = set.intersection(self._cube_methods,
            _sink_context_methods,
            _get_public_methods(slvr_cfg))

        if len(intersect) > 0:
            raise ValueError("'{i}' methods intersected on context"
                .format(i=intersect))

    @_setter_property
    def cube(self, value):
        self._cube = value

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        self._cfg = value

    @property
    def array(self):
        return self._array

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __getattr__(self, name):
        # Defer to the hypercube
        if name in self._cube_methods:
            return getattr(self._cube, name)
        # Avoid recursive calls to getattr
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(name)

_sink_context_methods = _get_public_methods(SinkContext)
