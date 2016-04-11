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

import pycuda.gpuarray as gpuarray
import pycuda.tools

from montblanc.node import Node

class RimeReduction(Node):
    def __init__(self):
        super(RimeReduction, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        # Run the kernel once so that its cached for use
        tmp_X2 = gpuarray.sum(slvr.chi_sqrd_result,
            stream=stream, allocator=slvr.dev_mem_pool.allocate)

        # Return the result's memory to the pool
        tmp_X2.gpudata.free()

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def execute(self, solver, stream=None):
        slvr = solver

        C = solver.const_data().cdata()

        # Call pycuda's internal reduction kernel on
        # the chi squared result array. We slice this array
        # with the problem subset specified in the
        # rime_const_data structure.
        # Note the returned result is a gpuarray
        # allocated with the supplied device memory pool
        X2_gpu_ary = gpuarray.sum(
            slvr.chi_sqrd_result[0:C.ntime.extents[1],
                0:C.nbl.extents[1], 0:C.nchan.extents[1]],
            stream=stream, allocator=slvr.dev_mem_pool.allocate)

        return X2_gpu_ary

    def post_execution(self, solver, stream=None):
        pass
