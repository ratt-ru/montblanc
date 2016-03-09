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

from montblanc.BaseSolver import BaseSolver

from montblanc.impl.biro.v5.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v5.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v5.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v5.gpu.RimeSumCoherencies import RimeSumCoherencies
from montblanc.impl.biro.v5.gpu.RimeReduction import RimeReduction

from montblanc.impl.biro.v4.BiroSolver import BiroSolver as BiroSolverV4

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

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

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            'E Beam cube width in l coords',
            slvr_cfg[Options.E_BEAM_WIDTH])

        self.register_dimension('beam_mh',
            'E Beam cube height in m coords',
            slvr_cfg[Options.E_BEAM_HEIGHT])

        self.register_dimension('beam_nud',
            'E Beam cube height in nu coords',
            slvr_cfg[Options.E_BEAM_DEPTH])

        wv = slvr_cfg[Options.WEIGHT_VECTOR]

        self.rime_e_beam = RimeEBeam()
        self.rime_b_sqrt = RimeBSqrt()
        self.rime_ekb_sqrt = RimeEKBSqrt()
        self.rime_sum = RimeSumCoherencies(weight_vector=wv)
        self.rime_reduce = RimeReduction()

        # Create a page-locked ndarray to hold constant GPU data
        # as well as
        # (1) A stream that this solver will asynchronously
        #     operate on
        # (2) An event indicating when an iteration of
        #     the kernels above have finished executing
        with self.context:
            self.const_data_buffer = cuda.pagelocked_empty(
                shape=mbu.rime_const_data_size(), dtype=np.int8)

            self.stream = cuda.Stream()
            self.kernels_done = cuda.Event()

        # Now create a cdata object wrapping the page-locked
        # ndarray and cast it to the rime_const_data c type.
        self.rime_const_data = mbu.wrap_rime_const_data(
            self.const_data_buffer)

        # Initialise it with the current solver (self)
        mbu.update_rime_const_data(self, self.rime_const_data)

        # Indicate these variables have not been set
        self.dev_mem_pool = None
        self.pinned_mem_pool = None

    def update_dimension(self, name, size=None,
        extents=None, safety=True):
        """
        Override update_dimension on BaseSolver.py to also
        update rime_const_data.
        """

        # Defer to parent method for house-keeping on the
        # solver, proper
        super(BiroSolver, self).update_dimension(name,
            size=size, extents=extents, safety=safety)

        # Update constant data
        mbu.update_rime_const_data(self, self.rime_const_data,
            sum_nsrc=True)

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
