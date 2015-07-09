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

from montblanc.impl.biro.v2.BiroSolver import BiroSolver as BiroSolverV2

from montblanc.impl.biro.v3.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v3.gpu.RimeGaussBSum import RimeGaussBSum

from montblanc.BaseSolver import BaseSolver

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

class BiroSolver(BaseSolver):
    """ Solver implementation for BIRO """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : BiroSolverConfiguration
                Solver Configuration variables
        """

        super(BiroSolver, self).__init__(slvr_cfg)

        wv = slvr_cfg.get(Options.WEIGHT_VECTOR, False)

        self.rime_ek = RimeEK()
        self.rime_b_sum = RimeGaussBSum(weight_vector=wv)

    def initialise(self):
        with self.context:
            self.rime_ek.initialise(self)
            self.rime_b_sum.initialise(self)

    def shutdown(self):
        with self.context:
            self.rime_ek.shutdown(self)
            self.rime_b_sum.shutdown(self)

    # Take these methods from the v2 BiroSolver
    get_default_base_ant_pairs = \
        BiroSolverV2.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BiroSolverV2.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BiroSolverV2.__dict__['get_ap_idx']
