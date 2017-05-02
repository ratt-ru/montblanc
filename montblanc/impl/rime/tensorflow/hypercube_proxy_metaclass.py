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

class HypercubeProxyMetaClass(type):
    """ MetaClass for classes that proxy HyperCubes """
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

            # wrap.__doc__ = (
            #     'def {}{}:\n'
            #     '\t""" {} """\n'
            #     '\treturn _proxy{}').format(name, fmt_args, method.__doc__, call_args)

            return wrap

        for name, method in hc_members:
            setattr(cls, name, wrap_cube_method(name, method.__func__))

        super(HypercubeProxyMetaClass, cls).__init__(name, bases, dct)
