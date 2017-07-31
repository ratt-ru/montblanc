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

import numpy as np

import hypercube as hc
from hypercube.array_factory import (
    create_local_arrays_on_cube,
    create_local_arrays,
    generic_stitch)

from rime_solver import RIMESolver

class MontblancNumpySolver(RIMESolver):
    def __init__(self, slvr_cfg):
        super(MontblancNumpySolver, self).__init__(slvr_cfg=slvr_cfg)

    def create_arrays(self, ignore=None, supplied=None):
        """
        Create any necessary arrays on the solver.

        Arguments
        ---------
            ignore : list
                List of array names to ignore.
            supplied : dictionary
                A dictionary of supplied arrays to create
                on the solver, keyed by name. Note that
                these arrays will not be initialised by
                montblanc, it is the responsibility of the
                user to initialise them.
        """
        if ignore is None:
            ignore = []

        if supplied is None:
            supplied = {}

        reified_arrays = self.arrays(reify=True)
        create_arrays = self._arrays_to_create(reified_arrays,
            ignore=ignore, supplied=supplied)

        # Create local arrays on the cube
        create_local_arrays_on_cube(self, create_arrays,
            array_stitch=generic_stitch,
            array_factory=np.empty)

        self._validate_supplied_arrays(reified_arrays, supplied)

        # Stitch the supplied arrays onto the cube
        generic_stitch(self, supplied)

        # Get our data source
        data_source = self._slvr_cfg[Options.DATA_SOURCE]

        # Initialise the arrays that we have created,
        # but not the supplied or ignored arrays
        for name, array in create_arrays.iteritems():
            cpu_ary = getattr(self, name)

            if data_source == Options.DATA_SOURCE_TEST:
                value = array.get(Options.DATA_SOURCE_TEST, None)
                self.init_array(name, cpu_ary, value)
            elif data_source == Options.DATA_SOURCE_EMPTY:
                pass
            else:
                self.init_array(name, cpu_ary,
                    array.get(Options.DATA_SOURCE_DEFAULT, None))