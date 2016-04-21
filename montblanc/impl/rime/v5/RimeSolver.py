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

from montblanc.solvers import MontblancCUDASolver

from montblanc.impl.rime.v5.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.rime.v5.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.rime.v5.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.rime.v5.gpu.RimeSumCoherencies import RimeSumCoherencies
from montblanc.impl.rime.v5.gpu.RimeReduction import RimeReduction

from montblanc.impl.rime.v4.RimeSolver import RimeSolver as RimeSolverV4

from montblanc.config import RimeSolverConfig as Options

class RimeSolver(MontblancCUDASolver):
    """ BIRO Solver Implementation """
    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        # Call the parent constructor
        super(RimeSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E cube nu depth')

        self.rime_e_beam = RimeEBeam()
        self.rime_b_sqrt = RimeBSqrt()
        self.rime_ekb_sqrt = RimeEKBSqrt()
        self.rime_sum = RimeSumCoherencies()
        self.rime_reduce = RimeReduction()

        from montblanc.impl.rime.v4.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        # Create
        # (1) A stream that this solver will asynchronously
        #     operate on
        # (2) An event indicating when an iteration of
        #     the kernels above have finished executing
        with self.context:
            self.stream = cuda.Stream()
            self.kernels_done = cuda.Event()

        # Create constant data for transfer to GPU
        self._const_data = mbu.create_rime_const_data(self)

        # Indicate these variables have not been set
        self._dev_mem_pool = None
        self._pinned_mem_pool = None
        self._pool_lock = None

    def const_data(self):
        return self._const_data

    def update_dimension(self, **update_dict):
        """
        Override update_dimension on HyperCube to also
        update rime_const_data.
        """

        # Defer to parent method on the base solver
        super(RimeSolver, self).update_dimension(**update_dict)

        # Update constant data, updating nsrc with sum of source counts
        self._const_data.update(self, sum_nsrc=True)

    @property
    def dev_mem_pool(self):
        return self._dev_mem_pool

    @dev_mem_pool.setter
    def dev_mem_pool(self, pool):
        self._dev_mem_pool = pool
    
    @property
    def pinned_mem_pool(self):
        return self._pinned_mem_pool

    @pinned_mem_pool.setter
    def pinned_mem_pool(self, pool):
        self._pinned_mem_pool = pool

    @property
    def pool_lock(self):
        return self._pool_lock

    @pool_lock.setter
    def pool_lock(self, lock):
        self._pool_lock = lock

    def initialise(self):
        with self.context:
            self.rime_e_beam.initialise(self)
            self.rime_b_sqrt.initialise(self)
            self.rime_ekb_sqrt.initialise(self)
            self.rime_sum.initialise(self)
            self.rime_reduce.initialise(self)

    def solve(self):
        with self.context:
            self.rime_e_beam.execute(self)
            self.rime_b_sqrt.execute(self)
            self.rime_ekb_sqrt.execute(self)
            self.rime_sum.execute(self)
            self.rime_reduce.execute(self)

    def shutdown(self):
        with self.context:
            self.rime_e_beam.shutdown(self)
            self.rime_b_sqrt.shutdown(self)
            self.rime_ekb_sqrt.shutdown(self)
            self.rime_sum.shutdown(self)
            self.rime_reduce.shutdown(self)