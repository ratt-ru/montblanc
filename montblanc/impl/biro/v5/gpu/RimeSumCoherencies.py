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
import pycuda.gpuarray as gpuarray

import montblanc.impl.biro.v4.gpu.RimeSumCoherencies

class RimeSumCoherencies(montblanc.impl.biro.v4.gpu.RimeSumCoherencies.RimeSumCoherencies):
    def __init__(self, weight_vector=False):
        super(RimeSumCoherencies, self).__init__(weight_vector=weight_vector)
    def initialise(self, solver, stream=None):
        super(RimeSumCoherencies, self).initialise(solver,stream)
    def shutdown(self, solver, stream=None):
        super(RimeSumCoherencies, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeSumCoherencies, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeSumCoherencies, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        if stream is not None:
            cuda.memcpy_htod_async(
                self.rime_const_data_gpu[0],
                slvr.const_data_buffer,
                stream=stream)
        else:
            cuda.memcpy_htod(
                self.rime_const_data_gpu[0],
                slvr.const_data_buffer)

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu

        sersic = np.intp(0) if np.product(slvr.sersic_shape_shape) == 0 \
            else slvr.sersic_shape_gpu

        self.kernel(slvr.uvw_gpu, gauss, sersic,
            slvr.frequency_gpu, slvr.ant_pairs_gpu,
            slvr.jones_gpu, slvr.weight_vector_gpu,
            slvr.bayes_data_gpu, slvr.G_term_gpu,
            slvr.vis_gpu, slvr.chi_sqrd_result_gpu,
            stream=stream, **self.launch_params)

        # Call the pycuda reduction kernel.
        # Divide by the single sigma squared value if a weight vector
        # is not required. Otherwise the kernel will incorporate the
        # individual sigma squared values into the sum
        #gpu_sum = gpuarray.sum(slvr.chi_sqrd_result_gpu).get()

        #if not self.weight_vector:
        #    slvr.set_X2(gpu_sum/slvr.sigma_sqrd)
        #else:
        #    slvr.set_X2(gpu_sum)
