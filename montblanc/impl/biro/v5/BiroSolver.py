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

from montblanc.impl.biro.v4.BiroSolver import BiroSolver as BiroSolverV4

from montblanc.impl.biro.v5.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v5.gpu.RimeGaussBSum import RimeGaussBSum

from montblanc.BaseSolver import BaseSolver
from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_NSSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

class BiroSolver(BaseSolver):
    """ Shared Data implementation for BIRO """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, nssrc=DEFAULT_NSSRC, dtype=DEFAULT_DTYPE,
        pipeline=None, **kwargs):
        """
        BiroSolver Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            nssrc : integer
                Number of sersic sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
            pipeline : list of nodes
                nodes defining the GPU kernels used to solve this RIME
        Keyword Arguments:
            context : pycuda.drivers.Context
                CUDA context to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSolver object.
        """

        # Turn off auto_correlations
        kwargs['auto_correlations'] = False

        super(BiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, nssrc=nssrc, dtype=dtype, pipeline=pipeline, **kwargs)

        self.rime_ek = RimeEK()
        self.rime_b_sum = RimeGaussBSum(weight_vector=kwargs.get('weight_vector', False))

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
        BiroSolverV4.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BiroSolverV4.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BiroSolverV4.__dict__['get_ap_idx']
