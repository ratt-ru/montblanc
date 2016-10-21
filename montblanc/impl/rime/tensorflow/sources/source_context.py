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
import inspect
import types

from hypercube import HyperCube

from montblanc.impl.rime.tensorflow.context_help import context_help

class _setter_property(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)

class SourceContextMetaClass(type):
    """ MetaClass for Sink Contexts """
    def __init__(cls, name, bases, dct):
        """ Proxy public methods on the HyperCube """
        def public_member_predicate(m):
            return inspect.ismethod(m) and not m.__name__.startswith('_')

        hc_members = inspect.getmembers(HyperCube, public_member_predicate)
        sc_members = inspect.getmembers(cls, public_member_predicate)

        intersect = set.intersection(
            set(m[0] for m in hc_members),
            set(m[0] for m in sc_members))

        if len(intersect) > 0:
            raise ValueError("Proxying methods failed on class '{c}'. "
                "The following members '{m}' conflicted with class '{hc}'."
                    .format(c=cls.__name__, m=list(intersect), hc=HyperCube.__name__))

        def wrap_cube_method(name, method):
            def _proxy(self, *args, **kwargs):
                return getattr(self._cube, name)(*args, **kwargs)

            wrap = functools.update_wrapper(_proxy, method)
            spec = inspect.getargspec(method)
            fmt_args = inspect.formatargspec(formatvalue=lambda v: '=_default', *spec)
            call_args = inspect.formatargspec(formatvalue=lambda v: '', *spec)

            wrap.__doc__ = (
                'def {}{}:\n'
                '\t""" {} """\n'
                '\treturn _proxy{}').format(name, fmt_args, method.__doc__, call_args)

            return wrap

        for name, method in hc_members:
            setattr(cls, name, wrap_cube_method(name, method.__func__))

        super(SourceContextMetaClass, cls).__init__(name, bases, dct)

class SourceContext(object):
    """
    Context for queue arrays.

    Proxies attributes of a hypercube and provides access to configuration
    """
    __slots__ = ('_cube', '_cfg', '_name', '_shape', '_dtype',
        '_iter_args', '_array_schema', '_cube_attributes')

    __metaclass__ = SourceContextMetaClass

    def __init__(self, name, cube, slvr_cfg, iter_args, array_schema, shape, dtype):
        self._name = name
        self._cube = cube
        self._cfg = slvr_cfg
        self._iter_args = iter_args
        self._array_schema = array_schema
        self._shape = shape
        self._dtype = dtype

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
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def array_schema(self):
        return self._array_schema

    @property
    def iter_args(self):
        return self._iter_args

    def help(self, display_cube=False):
        return context_help(self, display_cube)