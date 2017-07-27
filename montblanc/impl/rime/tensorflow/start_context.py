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


from .context_help import context_help
from .hypercube_proxy_metaclass import HypercubeProxyMetaClass

class StartContext(object):
    """
    Start Context object passed to Providers.

    It provides information to the user implementing a data source
    about the extents of the data tile that should be provided.

    .. code-block:: python

        # uvw varies by time and baseline and has 3 coordinate components
        cube.register_array("uvw", ("ntime", "nbl", 3), np.float64)

        ...

        class CustomSourceProvider(SourceProvider):
            def start(self, start_context):
                # Query dimensions directly
                (lt, ut), (lb, ub) = context.dim_extents("ntime", "nbl")
                ...

    Public methods of a :py:class:`~hypercube.base_cube.HyperCube`
    are proxied on this object. Other useful information, such
    as the configuration, iteration space arguments are also
    present on this object.
    """
    __slots__ = ('_cube', '_cfg', '_iter_args')

    __metaclass__ = HypercubeProxyMetaClass

    def __init__(self, cube, slvr_cfg, iter_args):
        self._cube = cube
        self._cfg = slvr_cfg
        self._iter_args = iter_args

    @property
    def cube(self):
        return self._cube

    @property
    def cfg(self):
        """
        Configuration
        """
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

    def help(self, display_cube=False):
        """
        Get help associated with this context

        Args
        -----
        display_cube: bool
            Add hypercube description to the output
        Returns
        -------
            str
                A help string associated with this context
        """
        return context_help(self, display_cube)