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

import hypercube as hc

from rime_solver import RIMESolver
from montblanc.config import SolverConfig as Options

class MontblancNumpySolver(RIMESolver):
    def __init__(self, slvr_cfg):
        super(MontblancNumpySolver, self).__init__(slvr_cfg=slvr_cfg)

    def create_arrays(self):
        # Create the local numpy arrays on ourself
        hc.create_local_numpy_arrays_on_cube(self)

        # Get our data source
        data_source = self._slvr_cfg[Options.DATA_SOURCE]

        for name, array in self.arrays().iteritems():
            cpu_ary = getattr(self, name)

            if data_source == Options.DATA_SOURCE_TEST:
                value = array.get(Options.DATA_SOURCE_TEST, None)
                self.init_array(name, cpu_ary, value)
            elif data_source == Options.DATA_SOURCE_EMPTY:
                pass
            else:
                self.init_array(name, cpu_ary,
                    array.get(Options.DATA_SOURCE_DEFAULT, None))    