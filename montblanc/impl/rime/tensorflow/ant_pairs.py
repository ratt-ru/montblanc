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

def default_ant_pairs(self):
    """
    Return a list of 2 arrays of shape (ntime, nbl)
    containing the default antenna pairs for each timestep
    at each baseline.
    """

    # Create the antenna pair mapping, from upper triangle indices
    # based on the number of antenna. Clamp this to the actual
    # number of baselines
    ntime, nbl = self.dim_local_size('ntime', 'nbl')
    ant0, ant1 = (ant[0:nbl] for ant in self.default_base_ant_pairs())

    return (
        np.tile(ant0, ntime).reshape(ntime, nbl),
        np.tile(ant1, ntime).reshape(ntime, nbl))

def monkey_patch_antenna_pairs(slvr):
    # Monkey patch these functions onto the solver object
    import types

    slvr.default_ant_pairs = types.MethodType(
        default_ant_pairs, slvr)
