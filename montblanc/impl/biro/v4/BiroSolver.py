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

import montblanc
import montblanc.util as mbu

from montblanc.solvers import MontblancCUDASolver
from montblanc.config import BiroSolverConfig as Options

from montblanc.impl.biro.v4.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v4.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v4.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v4.gpu.RimeSumCoherencies import RimeSumCoherencies

from montblanc.pipeline import Pipeline

def get_pipeline(slvr_cfg):
    return Pipeline([RimeBSqrt(),
        RimeEBeam(),
        RimeEKBSqrt(),
        RimeSumCoherencies()])

class BiroSolver(MontblancCUDASolver):
    """ BIRO Solver Implementation """

    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """

        # Set up a default pipeline if None is supplied
        slvr_cfg.setdefault('pipeline', get_pipeline(slvr_cfg))

        super(BiroSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube width in l coords')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube height in m coords')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube height in nu coords')

        # Monkey patch these functions onto the object
        # TODO: Remove this when deprecating v2.
        from montblanc.impl.biro.v4.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)
   
        from montblanc.impl.biro.v4.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        self._const_data = mbu.create_rime_const_data(self, self.context)

    def const_data(self):
        return self._const_data
