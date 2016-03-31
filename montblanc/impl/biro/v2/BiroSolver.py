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

from montblanc.solvers import MontblancCUDASolver
from montblanc.config import BiroSolverConfig as Options

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum
from montblanc.pipeline import Pipeline

def get_pipeline(slvr_cfg):
    wv = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
    return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=wv)])

class BiroSolver(MontblancCUDASolver):
    """ Solver implementation for BIRO """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : BiroSolverConfiguration
        """

        # Set up a default pipeline if None is supplied
        slvr_cfg.setdefault('pipeline', get_pipeline(slvr_cfg))

        super(BiroSolver, self).__init__(slvr_cfg)

        # Monkey patch these functions onto the object
        # TODO: Remove this when deprecating v2.
        from ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        from montblanc.impl.biro.v2.config import (A, P)

        self.register_default_dimensions()
        self.register_properties(P)
        self.register_arrays(A)  

    def solve(self):
        """ Solve the RIME """
        with self.context as ctx:
            self.pipeline.execute(self)

    def initialise(self):
        with self.context as ctx:
            self.pipeline.initialise(self)

    def shutdown(self):
        """ Stop the RIME solver """
        with self.context as ctx:
            self.pipeline.shutdown(self)        


