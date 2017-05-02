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

from ..context_help import context_help
from ..hypercube_proxy_metaclass import HypercubeProxyMetaClass

class _setter_property(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)

class SinkContext(object):
    """
    Context object passed to data sinks.

    Primarily, it exists to provide a tile of output data to the user.

    .. code-block:: python

        class MySinkProvider(SinkProvider):
            vis_queue = Queue(10)

            ...
            def model_vis(self, context):
                print context.help(display_cube=True)
                # Consume data
                vis_queue.put(context.data)


    Public methods of a :py:class:`~hypercube.base_cube.HyperCube`
    are proxied on this object. Other useful information, such
    as the configuration, iteration space arguments and the
    abstract array schema are also present on this object.
    """

    __slots__ = ('_cube', '_cfg', '_name', '_data', '_input_cache',
        '_cube_attributes', '_iter_args', '_array_schema')

    __metaclass__ = HypercubeProxyMetaClass

    def __init__(self, name, cube, slvr_cfg,
            iter_args, array_schema,
            data, input_cache):

        self._name = name
        self._cube = cube
        self._iter_args = iter_args
        self._array_schema = array_schema
        self._cfg = slvr_cfg
        self._data = data
        self._input_cache = input_cache

    @_setter_property
    def cube(self, value):
        self._cube = value

    @property
    def cfg(self):
        """ Configuration """
        return self._cfg

    @cfg.setter
    def cfg(self, value):
        self._cfg = value

    @property
    def iter_args(self):
        """
        Iteration arguments that describe the tile sizes
        over which iteration is performed. In the following example,
        iteration is occuring in tiles of 100 Timesteps, 64 Channels
        and 50 Point Sources.

        .. code-block:: python

            context.iter_args == [("ntime", 100),
                    ("nchan", 64), ("npsrc", 50)]
        """
        return self._iter_args

    @property
    def array_schema(self):
        """
        The array schema of the array associated
        with this data source. For instance if `model_vis` is
        registered on a hypercube as follows:

        .. code-block:: python

            # Register model_vis array_schema on hypercube
            cube.register_array("model_vis",
                ("ntime", "nbl", "nchan", "ncorr"),
                np.complex128)

            ...
            # Create a source context for model_vis data source
            context = SourceContext("model_vis", ...)
            ...
            # Obtain the array schema
            context.array_schema == ("ntime", "nbl", "nchan", "ncorr")

        """
        return self._array_schema

    @property
    def data(self):
        """
        The data tile available for consumption by the associated sink
        """
        return self._data

    @property
    def input(self):
        """
        The dictionary of inputs used to produce
        :py:obj:`~SinkContext.data`. For example, if one
        wished to find the antenna pair used to produce a
        particular model visibility, one could do the following:

        .. code-block:: python

            def model_vis(self, context):
                ant1 = context.input["antenna1"]
                ant2 = context.input["antenna2"]
                model_vis = context.data

        """
        return self._input_cache

    @property
    def name(self):
        """ The name of the data sink of this context. """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def help(self, display_cube=False):
        """
        Get help associated with this context

        Parameters
        ----------
        display_cube: bool
            Add hypercube description to the output
        Returns
        -------
            str
                A help string associated with this context
        """
        return context_help(self, display_cube)
