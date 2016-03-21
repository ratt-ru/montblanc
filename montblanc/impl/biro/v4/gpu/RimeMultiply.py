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

import pycuda.gpuarray as gpuarray

import montblanc
import montblanc.util as mbu
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 1024,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 1
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 256,
    'BLOCKDIMY' : 1,
    'BLOCKDIMZ' : 1
}

KERNEL_TEMPLATE = string.Template("""
#include <cstdio>
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NPSRC ${npsrc}
#define NSRC ${nsrc}

#define NJONES NBL*NCHAN*NTIME*NSRC

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ void
complex_multiply(const typename Tr::ct * lhs, const typename Tr::ct * rhs, typename Tr::ct * result)
{
    result->x = lhs->x*rhs->x;
    result->y = lhs->x*rhs->y;
    result->x -= lhs->y*rhs->y; /* RE*RE - IM*IM */
    result->y += lhs->y*rhs->x; /* RE*IM + IM*RE */
}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ void
complex_multiply_add(const typename Tr::ct * lhs, const typename Tr::ct * rhs, typename Tr::ct * result)
{
    result->x += lhs->x*rhs->x;
    result->y += lhs->x*rhs->y;
    result->x -= lhs->y*rhs->y; /* RE*RE - IM*IM */
    result->y += lhs->y*rhs->x; /* RE*IM + IM*RE */
}

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_multiply_impl(
    typename Tr::ct * lhs,
    typename Tr::ct * rhs,
    typename Tr::ct * out_jones)
{
    __shared__ typename Tr::ct result[BLOCKDIMX];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= NJONES)
        { return; }

    const typename Tr::ct a00 = lhs[i]; i += NJONES;
    const typename Tr::ct a01 = lhs[i]; i += NJONES;
    const typename Tr::ct a10 = lhs[i]; i += NJONES;
    const typename Tr::ct a11 = lhs[i];

    i = blockIdx.x*blockDim.x + threadIdx.x;

    const typename Tr::ct b00 = rhs[i]; i += NJONES;
    const typename Tr::ct b01 = rhs[i]; i += NJONES;
    const typename Tr::ct b10 = rhs[i]; i += NJONES;
    const typename Tr::ct b11 = rhs[i];

    complex_multiply<T>(&a00,&b00,&result[threadIdx.x]);
    complex_multiply_add<T>(&a01,&b10,&result[threadIdx.x]);
    i = blockIdx.x*blockDim.x + threadIdx.x;
    out_jones[i] = result[threadIdx.x];

    complex_multiply<T>(&a00,&b01,&result[threadIdx.x]);
    complex_multiply_add<T>(&a01,&b11,&result[threadIdx.x]);
    i += NJONES;
    out_jones[i] = result[threadIdx.x];

    complex_multiply<T>(&a10,&b00,&result[threadIdx.x]);
    complex_multiply_add<T>(&a11,&b10,&result[threadIdx.x]);
    i += NJONES;
    out_jones[i] = result[threadIdx.x];

    complex_multiply<T>(&a10,&b01,&result[threadIdx.x]);
    complex_multiply_add<T>(&a11,&b11,&result[threadIdx.x]);
    i += NJONES;
    out_jones[i] = result[threadIdx.x];
}

extern "C" {

__global__
void rime_jones_multiply_float(
    float2 * lhs,
    float2 * rhs,
    float2 * out_jones)
{
    rime_jones_multiply_impl<float>(lhs, rhs, out_jones);
}

__global__
void rime_jones_multiply_double(
    double2 * lhs,
    double2 * rhs,
    double2 * out_jones)
{
    rime_jones_multiply_impl<double>(lhs, rhs, out_jones);
}

}

""")

class RimeMultiply(Node):
    def __init__(self):
        super(RimeMultiply, self).__init__()

    def initialise(self, solver):
        slvr = solver

        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS
        D.update(slvr.template_dict())

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo'],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        kname = 'rime_jones_multiply_float' \
            if slvr.is_float() is True else \
            'rime_jones_multiply_double'

        self.kernel = self.mod.get_function(kname)

    def shutdown(self, solver):
        pass

    def pre_execution(self, solver):
        pass

    def get_kernel_params(self, solver):
        slvr = solver
        nsrc, ntime, nbl, nchan = slvr.dim_local_size(
            'nsrc', 'ntime', 'nbl', 'nchan')
        D = FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS

        njones = nbl*nchan*ntime*nsrc
        jones_per_block = D['BLOCKDIMX'] if njones > D['BLOCKDIMX'] else njones
        jones_blocks = mbu.blocks_required(njones,jones_per_block)

        return {
            'block'  : (jones_per_block,1,1), \
            'grid'   : (jones_blocks,1,1)
        }

    def execute(self, solver):
        slvr = solver
        nsrc, ntime, nbl, nchan = slvr.dim_local_size(
            'nsrc', 'ntime', 'nbl', 'nchan')

        # Output jones matrix
        njones = nbl*nchan*ntime*nsrc
        jsize = np.product(slvr.jones_shape) # Number of complex  numbers

        # TODO: This is all wrong and should be replaced with actual gpuarray's
        # living on the slvr object
        jones_rhs = (np.random.random(jsize) + \
             1j*np.random.random(jsize)) \
            .astype(np.complex128).reshape(slvr.jones_shape)

        jones_lhs_gpu = slvr.jones_gpu
        jones_rhs_gpu = gpuarray.to_gpu(jones_rhs)
        jones_output_gpu = gpuarray.empty(shape=slvr.jones_shape, dtype=np.complex128)

        self.kernel(jones_lhs_gpu, jones_rhs_gpu, jones_output_gpu, \
            **self.get_kernel_params(slvr))

        slvr.jones_gpu = jones_output_gpu

    def post_execution(self, solver):
        pass
