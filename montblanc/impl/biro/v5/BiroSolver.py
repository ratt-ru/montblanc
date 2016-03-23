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
import pycuda.driver as cuda

import montblanc
import montblanc.util as mbu

from montblanc.base_solver import BaseSolver

from montblanc.impl.biro.v5.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v5.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v5.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v5.gpu.RimeSumCoherencies import RimeSumCoherencies
from montblanc.impl.biro.v5.gpu.RimeReduction import RimeReduction

from montblanc.impl.biro.v4.BiroSolver import BiroSolver as BiroSolverV4

from montblanc.config import BiroSolverConfig as Options

class BiroSolver(BaseSolver):
    """ BIRO Solver Implementation """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        # Call the parent constructor
        super(BiroSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension(Options.E_BEAM_WIDTH,
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube width in l coords')

        self.register_dimension(Options.E_BEAM_HEIGHT,
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube height in m coords')

        self.register_dimension(Options.E_BEAM_DEPTH,
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube height in nu coords')
        wv = slvr_cfg[Options.WEIGHT_VECTOR]

        self.rime_e_beam = RimeEBeam()
        self.rime_b_sqrt = RimeBSqrt()
        self.rime_ekb_sqrt = RimeEKBSqrt()
        self.rime_sum = RimeSumCoherencies(weight_vector=wv)
        self.rime_reduce = RimeReduction()

        # Create
        # (1) A stream that this solver will asynchronously
        #     operate on
        # (2) An event indicating when an iteration of
        #     the kernels above have finished executing
        with self.context:
            self.stream = cuda.Stream()
            self.kernels_done = cuda.Event()

        # Create constant data for transfer to GPU
        self._const_data = mbu.create_rime_const_data(self, self.context)

        # Indicate these variables have not been set
        self.dev_mem_pool = None
        self.pinned_mem_pool = None

    def const_data(self):
        return self._const_data

    def update_dimension(self, dim_data):
        """
        Override update_dimension on BaseSolver.py to also
        update rime_const_data.
        """

        # Defer to parent method on the base solver
        super(BiroSolver, self).update_dimension(dim_data)

        # Update constant data, updating nsrc with sum of source counts
        self._const_data.update(self, sum_nsrc=True)

    def set_dev_mem_pool(self, dev_mem_pool):
        self.dev_mem_pool = dev_mem_pool

    def set_pinned_mem_pool(self, pinned_mem_pool):
        self.pinned_mem_pool = pinned_mem_pool

    def initialise(self):
        with self.context:
            self.rime_e_beam.initialise(self)
            self.rime_b_sqrt.initialise(self)
            self.rime_ekb_sqrt.initialise(self)
            self.rime_sum.initialise(self)
            self.rime_reduce.initialise(self)

    def shutdown(self):
        with self.context:
            self.rime_e_beam.shutdown(self)
            self.rime_b_sqrt.shutdown(self)
            self.rime_ekb_sqrt.shutdown(self)
            self.rime_sum.shutdown(self)
            self.rime_reduce.shutdown(self)

    # Take these methods from the v4 BiroSolver
    get_default_base_ant_pairs = \
        BiroSolverV4.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BiroSolverV4.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BiroSolverV4.__dict__['get_ap_idx']
