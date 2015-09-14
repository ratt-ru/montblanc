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

        super(BiroSolver, self).__init__(slvr_cfg)

        # Configure the dimensions of the beam cube
        self.beam_lw = self.slvr_cfg[Options.E_BEAM_WIDTH]
        self.beam_mh = self.slvr_cfg[Options.E_BEAM_HEIGHT]
        self.beam_nud = self.slvr_cfg[Options.E_BEAM_DEPTH]

        wv = slvr_cfg[Options.WEIGHT_VECTOR]

        self.rime_e_beam = RimeEBeam()
        self.rime_b_sqrt = RimeBSqrt()
        self.rime_ekb_sqrt = RimeEKBSqrt()
        self.rime_sum = RimeSumCoherencies(weight_vector=wv)
        self.rime_reduce = RimeReduction()

        # Create a page-locked ndarray to hold constant GPU data
        with self.context:
            self.const_data_buffer = cuda.pagelocked_empty(
                shape=mbu.rime_const_data_size(), dtype=np.int8)

        # Now create a cdata object wrapping the page-locked
        # ndarray and cast it to the rime_const_data c type.
        self.rime_const_data = mbu.wrap_rime_const_data(
            self.const_data_buffer)

        # Initialise it with the current solver (self)
        mbu.init_rime_const_data(self, self.rime_const_data)

        # Indicate these variables have not been set
        self.dev_mem_pool = None
        self.pinned_mem_pool = None

    def cfg_total_src_dims(self, nsrc):
        """
        Configure the total number of sources that will
        be handled by this solver. Used by v5 to allocate
        solvers handling subsets of the total problem.
        Passing nsrc=100 means that the solver will handle
        100 sources in total.

        Additionally, sets the number for each individual
        source type to 100. So npsrc=100, ngsrc=100,
        nssrc=100 for instance. This is because if we're
        handling 100 sources total, we'll need space for
        at least 100 sources of each type.

        The number of sources actually handled by the
        solver on each iteration is set in the
        rime_const_data_cpu structure.

        """
        self.nsrc = nsrc
        
        for nr_var in mbu.source_nr_vars():
            setattr(self, nr_var, nsrc)

    def cfg_sub_dims(self, counts):
        """
        Configure the dimensions of the subset of the
        RIME solved by this solvers.

        Sets these dimensions on the rime_const_data_cpu
        structure, which is passed to the kernels on each
        run.
        """

        # Set key-value pairs on rime_const_data
        # from kwargs
        for key, value in counts.iteritems():
            if hasattr(self.rime_const_data, key):
                setattr(self.rime_const_data, key, value)
            else:
                montblanc.log.warn((
                    'Attempted to set %s=%s '
                    'on rime_const_data but key %s '
                    'is not present') % (key, value, key))

        # Set rime_const_data.nsrc by summing
        # each source type
        setattr(self.rime_const_data, 'nsrc',
            sum([getattr(self.rime_const_data, s)
                for s in mbu.source_nr_vars()]))


    def get_properties(self):
        # Obtain base solver property dictionary
        # and add the beam cube dimensions to it
        D = super(BiroSolver, self).get_properties()

        D.update({
            Options.E_BEAM_WIDTH : self.beam_lw,
            Options.E_BEAM_HEIGHT : self.beam_mh,
            Options.E_BEAM_DEPTH : self.beam_nud
        })

        return D

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
