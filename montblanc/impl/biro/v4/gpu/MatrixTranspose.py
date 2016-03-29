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
import string

from pycuda.compiler import SourceModule

import montblanc
import montblanc.util as mbu
from montblanc.node import Node

FLOAT_PARAMS = {
    'TILE_DIM': 32,
    'BLOCKDIMX': 32,
    'BLOCKDIMY': 4,
    'BLOCKDIMZ': 1,
    'maxregs': 63        # Maximum number of registers
}

DOUBLE_PARAMS = {
    'TILE_DIM': 32,
    'BLOCKDIMX': 32,
    'BLOCKDIMY': 4,
    'BLOCKDIMZ': 1,
    'maxregs': 63         # Maximum number of registers
}

# Based on the following post
# http://arrayfire.com/cuda-optimization-tips-for-matrix-transpose-in-real-world-applications/
# which in turn is based on
# http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
KERNEL_TEMPLATE = string.Template("""
#define TILE_DIM ${TILE_DIM}

template<typename T, bool is32Multiple>
__device__
void transposeSC(const T * in, T * out,  unsigned int nx, unsigned int ny)
{
    __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];

    unsigned lx = threadIdx.x;
    unsigned ly = threadIdx.y;

    unsigned gx = lx + blockDim.x * blockIdx.x;
    unsigned gy = ly + TILE_DIM   * blockIdx.y;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<nx && gy_<ny))
            shrdMem[ly + repeat][lx] = in[gy_ * nx + gx];
    }

    __syncthreads();

    gx = lx + blockDim.x * blockIdx.y;
    gy = ly + TILE_DIM   * blockIdx.x;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<ny && gy_<nx))
            out[gy_ * ny + gx] = shrdMem[lx][ly + repeat];
    }
}

extern "C" {
// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
// Arguments
// - ft: The floating point type. Should be float/double.
#define stamp_matrix_transpose_fn(ft) \
__global__ \
void matrix_transpose_ ## ft( \
    const ft * in, \
    ft * out, \
    unsigned int nx, unsigned int ny) \
{ \
    transposeSC<ft, false>(in, out, nx, ny); \
}

stamp_matrix_transpose_fn(float)
stamp_matrix_transpose_fn(double)
stamp_matrix_transpose_fn(float2)
stamp_matrix_transpose_fn(double2)

} // extern "C" {

""")

class MatrixTranspose(Node):
    def __init__(self):
        super(MatrixTranspose, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        D = slvr.template_dict()
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'matrix_transpose_' + \
            ('float' if slvr.is_float() is True else 'double')

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def get_kernel_params(self, solver, nx, ny):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        #x_per_block = D['BLOCKDIMX'] if nx > D['BLOCKDIMX'] else nx
        #y_per_block = D['BLOCKDIMY'] if ny > D['BLOCKDIMX'] else ny
        TILE_DIM = D['TILE_DIM']
        x_per_block = D['BLOCKDIMX']
        y_per_block = D['BLOCKDIMY']

        x_blocks = mbu.blocks_required(nx, TILE_DIM)
        y_blocks = mbu.blocks_required(ny, TILE_DIM)

        return {
            'block' : (x_per_block, y_per_block,1),
            'grid'  : (x_blocks, y_blocks, 1),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        ny, nx = slvr.matrix_in_gpu.shape
        params = self.get_kernel_params(slvr, nx, ny)

        print 'Transposing'

        self.kernel(slvr.matrix_in_gpu, slvr.matrix_out_gpu,
            np.int32(nx), np.int32(ny), **params)

    def post_execution(self, solver, stream=None):
        pass
