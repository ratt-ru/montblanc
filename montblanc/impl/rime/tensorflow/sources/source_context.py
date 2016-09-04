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

def _get_public_attributes(obj):
	""" Return the public attributes on an object """
	return set(n for n, m in inspect.getmembers(obj) if not n.startswith('_'))

class _setter_property(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)

class SourceContext(object):
	"""
	Context for queue arrays.

	Proxies attributes of a hypercube and provides access to configuration
	"""
	__slots__ = ('_cube', '_cfg', '_name', '_shape', '_dtype',
		'_iter_args', '_array_schema', '_cube_attributes')

	def __init__(self, name, cube, slvr_cfg, iter_args, array_schema, shape, dtype):
		self._name = name
		self._cube = cube
		self._cfg = slvr_cfg
		self._iter_args = iter_args
		self._array_schema = array_schema
		self._shape = shape
		self._dtype = dtype

        # TODO: Only do _get_public_attributes once for
        # cube and slvr_cfg below. It's probably enough
        # to do this at a class level and SourceContexts
        # are created fairly often
		self._cube_attributes = _get_public_attributes(cube)

		# Fall over if there's any intersection between the
		# public attributes the current class and the
		# union of the hypercube and the configuration attributes
		intersect = set.intersection(_source_context_attributes,
			set.union(self._cube_attributes, _get_public_attributes(slvr_cfg)))

		if len(intersect) > 0:
			raise ValueError("'{i}' attributes intersected on context"
				.format(i=list(intersect)))

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

	def __getattr__(self, name):
		# Defer to the hypercube
		if name in self._cube_attributes:
			return getattr(self._cube, name)
		# Avoid recursive calls to getattr
		elif hasattr(self, name):
			return getattr(self, name)
		else:
			raise AttributeError(name)

_source_context_attributes = _get_public_attributes(SourceContext)
