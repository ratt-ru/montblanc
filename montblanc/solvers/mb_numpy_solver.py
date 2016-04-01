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

from hypercube import NumpyHyperCube
from rime_solver import RIMESolver
from montblanc.config import SolverConfig as Options

class MontblancNumpySolver(RIMESolver, NumpyHyperCube):
    def __init__(self, slvr_cfg):
        super(MontblancNumpySolver, self).__init__(slvr_cfg=slvr_cfg)

    def register_array(self, name, shape, dtype, **kwargs):
        """
        Register an array with this Solver object.

        Arguments
        ----------
            name : string
                name of the array.
            shape : integer/string or tuple of integers/strings
                Shape of the array.
            dtype : data-type
                The data-type for the array.

        Keyword Arguments
        -----------------
            page_locked : boolean
                True if the 'name' ndarray should be allocated as
                a page-locked array.
            aligned : boolean
                True if the 'name' ndarray should be allocated as
                an page-aligned array.

        Returns
        -------
            A dictionary describing this array.
        """

        A = super(MontblancNumpySolver, self).register_array(
            name, shape, dtype, **kwargs)

        # Our parent will create this
        cpu_ary = getattr(self, name)
        data_source = self._slvr_cfg[Options.DATA_SOURCE]

        # If we're creating test data, initialise the array with
        # data from the test key, don't initialise if we've been
        # explicitly told the array should be empty, otherwise
        # set the defaults
        if data_source == Options.DATA_SOURCE_TEST:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_TEST, None))
        elif data_source == Options.DATA_SOURCE_EMPTY:
            pass
        else:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_DEFAULT, None))

        return A