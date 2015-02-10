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

import string

from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX': 32,
    'BLOCKDIMY': 32,
    'BLOCKDIMZ': 1,
    'maxregs': 63        # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX': 32,
    'BLOCKDIMY': 32,
    'BLOCKDIMZ': 1,
    'maxregs': 63         # Maximum number of registers
}

# Taken from the following post
# http://arrayfire.com/cuda-optimization-tips-for-matrix-transpose-in-real-world-applications/
# which in turn is based on
# http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
KERNEL_TEMPLATE = string.Template("""
#define TILE_DIM 32

template<typename T, bool is32Multiple>
__device__
void transposeSC(T * out, const T * in, unsigned int dim0, unsigned int dim1)
{
    __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];

    unsigned lx = threadIdx.x;
    unsigned ly = threadIdx.y;

    unsigned gx = lx + blockDim.x * blockIdx.x;
    unsigned gy = ly + TILE_DIM   * blockIdx.y;

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<dim0 && gy_<dim1))
            shrdMem[ly + repeat][lx] = in[gy_ * dim0 + gx];
    }

    __syncthreads();

    gx = lx + blockDim.x * blockIdx.y;
    gy = ly + TILE_DIM   * blockIdx.x;

     //same code as transpose32 kernel

#pragma unroll
    for (unsigned repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
        unsigned gy_ = gy+repeat;
        if (is32Multiple || (gx<dim1 && gy_<dim0))
            out[gy_ * dim0 + gx] = shrdMem[lx][ly + repeat];
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
    ft * in, \
    const ft * out, \
    unsigned int dim0, unsigned int dim1) \
{ \
    transposeSC<ft, false>(in, out, dim0, dim1); \
}

stamp_matrix_transpose_fn(float)
stamp_matrix_transpose_fn(double)

} // extern "C" {

""")

class MatrixTranspose(Node):
    def __init__(self):
        super(MatrixTranspose, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        D = slvr.get_properties()
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

    def get_kernel_params(self, solver):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        chans_per_block = D['BLOCKDIMX'] if slvr.nchan > D['BLOCKDIMX'] else slvr.nchan
        bl_per_block = D['BLOCKDIMY'] if slvr.nbl > D['BLOCKDIMY'] else slvr.nbl
        times_per_block = D['BLOCKDIMZ'] if slvr.ntime > D['BLOCKDIMZ'] else slvr.ntime

        chan_blocks = self.blocks_required(slvr.nchan, chans_per_block)
        bl_blocks = self.blocks_required(slvr.nbl, bl_per_block)
        time_blocks = self.blocks_required(slvr.ntime, times_per_block)

        return {
            'block' : (chans_per_block, bl_per_block, times_per_block),
            'grid'  : (chan_blocks, bl_blocks, time_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

    def post_execution(self, solver, stream=None):
        pass
