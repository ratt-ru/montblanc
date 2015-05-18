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

import pycuda.driver
import pycuda.tools

import montblanc.impl.biro.v2.gpu.RimeGaussBSum

class RimeGaussBSum(montblanc.impl.biro.v2.gpu.RimeGaussBSum.RimeGaussBSum):
    def __init__(self, weight_vector=False):
        super(RimeGaussBSum, self).__init__(weight_vector)
    def initialise(self, solver, stream=None):
        super(RimeGaussBSum, self).initialise(solver,stream)
    def shutdown(self, solver, stream=None):
        super(RimeGaussBSum, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeGaussBSum, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeGaussBSum, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu
        sersic = np.intp(0) if np.product(slvr.sersic_shape_shape) == 0 \
            else slvr.sersic_shape_gpu

        self.kernel(slvr.uvw_gpu, slvr.brightness_gpu, gauss, sersic,
            slvr.wavelength_gpu, slvr.ant_pairs_gpu, slvr.jones_scalar_gpu,
            slvr.weight_vector_gpu, slvr.vis_gpu, slvr.bayes_data_gpu,
            slvr.chi_sqrd_result_gpu,
            stream=stream, **self.get_kernel_params(slvr))
