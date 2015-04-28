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

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import montblanc
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX': 32,   # Number of channels*4 polarisations
    'BLOCKDIMY': 8,     # Number of baselines
    'BLOCKDIMZ': 1,     # Number of timesteps
    'maxregs': 48         # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX': 32,   # Number of channels*4 polarisations
    'BLOCKDIMY': 8,     # Number of baselines
    'BLOCKDIMZ': 1,     # Number of timesteps
    'maxregs': 48         # Maximum number of registers
}

KERNEL_TEMPLATE = string.Template("""
#include <cstdio>
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>
#include <montblanc/include/brightness.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NPSRC ${npsrc}
#define NGSRC ${ngsrc}
#define NSSRC ${nssrc}
#define NSRC ${nsrc}

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

#define GAUSS_SCALE ${gauss_scale}
#define TWO_PI ${two_pi}

template <
    typename T,
    bool apply_weights=false,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_gauss_B_sum_impl(
    typename Tr::ft * uvw,
    typename Tr::ft * stokes,
    typename Tr::ft * gauss_shape,
    typename Tr::ft * sersic_shape,
    typename Tr::ft * wavelength,
    int * ant_pairs,
    typename Tr::ct * jones_EK_scalar,
    typename Tr::ft * weight_vector,
    typename Tr::ct * visibilities,
    typename Tr::ct * data_vis,
    typename Tr::ft * chi_sqrd_result)
{
    int CHAN = (blockIdx.x*blockDim.x + threadIdx.x)>>2;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    #define NPOL 4
    #define POL (threadIdx.x & 0x3)

    if(BL >= NBL || TIME >= NTIME || CHAN >= NCHAN)
        return;

    volatile __shared__ T shuvw[BLOCKDIMZ][BLOCKDIMY][3];

    volatile __shared__ T wl[BLOCKDIMX];

    int i;

    // Figure out the antenna pairs
    i = TIME*NBL + BL;   int ANT1 = ant_pairs[i];
    i += NBL*NTIME;      int ANT2 = ant_pairs[i];

    // UVW coordinates vary by baseline and time, but not channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = TIME*NA + ANT1;    shuvw[threadIdx.z][threadIdx.y][0] = uvw[i];
        i += NA*NTIME;         shuvw[threadIdx.z][threadIdx.y][1] = uvw[i];
        i += NA*NTIME;         shuvw[threadIdx.z][threadIdx.y][2] = uvw[i];

        i = TIME*NA + ANT2;    shuvw[threadIdx.z][threadIdx.y][0] -= uvw[i];
        i += NA*NTIME;         shuvw[threadIdx.z][threadIdx.y][1] -= uvw[i];
        i += NA*NTIME;         shuvw[threadIdx.z][threadIdx.y][2] -= uvw[i];
    }

    // Wavelength varies by channel, but not baseline and time
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { wl[threadIdx.x] = wavelength[CHAN]; }

    typename Tr::ct polsum = Po::make_ct(0.0, 0.0);

    for(int SRC=0;SRC<NPSRC;++SRC)
    {
        i = TIME*NSRC + SRC + POL;
        typename Tr::ft pol = stokes[i];
        typename Tr::ct brightness;
        montblanc::create_brightness<T>(brightness, pol);

        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NCHAN + CHAN + POL;
        typename Tr::ct ant_one = jones_EK_scalar[i]; ant_one.y = -ant_one.y;
        montblanc::complex_multiply_in_place<T>(brightness, ant_one);
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NCHAN + CHAN + POL;
        typename Tr::ct ant_two = jones_EK_scalar[i];
        montblanc::complex_multiply_in_place<T>(ant_two, brightness);

        polsum.x += ant_two.x;
        polsum.y += ant_two.y;

        __syncthreads();
    }

    for(int SRC=NPSRC;SRC<NPSRC+NGSRC;++SRC)
    {
        i = TIME*NSRC + SRC + POL;
        typename Tr::ft pol = stokes[i];
        typename Tr::ct brightness;
        montblanc::create_brightness<T>(brightness, pol);

        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NCHAN + CHAN + POL;
        typename Tr::ct ant_one = jones_EK_scalar[i]; ant_one.y = -ant_one.y;
        montblanc::complex_multiply_in_place<T>(brightness, ant_one);
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NCHAN + CHAN + POL;
        typename Tr::ct ant_two = jones_EK_scalar[i];
        montblanc::complex_multiply_in_place<T>(ant_two, brightness);

        polsum.x += ant_two.x;
        polsum.y += ant_two.y;

        __syncthreads();
    }

    i = (TIME*NBL + BL)*NCHAN + CHAN + POL;
    visibilities[i] = polsum;
    typename Tr::ct delta = data_vis[i];
    delta.x -= polsum.x; delta.y -= polsum.y;
    delta.x *= delta.x; delta.y *= delta.y;
    if(apply_weights) { T w = weight_vector[i]; delta.x *= w; delta.y *= w; }

    i = (TIME*NBL + BL)*NCHAN + CHAN;
    chi_sqrd_result[i] = delta.x + delta.y;
}

extern "C" {

// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - apply_weights: boolean indicating whether we're weighting our visibilities
// - symbol: u or w depending on whether we're handling unweighted/weighted visibilities.

#define stamp_gauss_b_sum_fn(ft, ct, apply_weights, symbol) \
__global__ void \
rime_gauss_B_sum_ ## symbol ## chi_ ## ft( \
    ft * uvw, \
    ft * stokes, \
    ft * gauss_shape, \
    ft * sersic_shape, \
    ft * wavelength, \
    int * ant_pairs, \
    ct * jones_EK_scalar, \
    ft * weight_vector, \
    ct * visibilities, \
    ct * data_vis, \
    ft * chi_sqrd_result) \
{ \
    rime_gauss_B_sum_impl<ft, apply_weights>(uvw, stokes, gauss_shape, sersic_shape, \
        wavelength, ant_pairs, jones_EK_scalar, \
        weight_vector, visibilities, data_vis, \
        chi_sqrd_result); \
}

stamp_gauss_b_sum_fn(float, float2, false, u)
stamp_gauss_b_sum_fn(float, float2, true, w)
stamp_gauss_b_sum_fn(double, double2, false, u)
stamp_gauss_b_sum_fn(double, double2, true, w)

} // extern "C" {
""")

class RimeGaussBSum(Node):
    def __init__(self, weight_vector=False):
        super(RimeGaussBSum, self).__init__()
        self.weight_vector = weight_vector

    def initialise(self, solver, stream=None):
        slvr = solver

        D = slvr.get_properties()
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_gauss_B_sum_' + \
            ('w' if self.weight_vector else 'u') + 'chi_' + \
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

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu

        sersic = np.intp(0) if np.product(slvr.sersic_shape_shape) == 0 \
            else slvr.sersic_shape_gpu

        self.kernel(slvr.uvw_gpu, slvr.stokes_gpu, gauss, sersic,
            slvr.wavelength_gpu, slvr.ant_pairs_gpu, slvr.jones_scalar_gpu,
            slvr.weight_vector_gpu,
            slvr.vis_gpu, slvr.bayes_data_gpu, slvr.chi_sqrd_result_gpu,
            **self.get_kernel_params(slvr))

        # Call the pycuda reduction kernel.
        # Divide by the single sigma squared value if a weight vector
        # is not required. Otherwise the kernel will incorporate the
        # individual sigma squared values into the sum
        gpu_sum = gpuarray.sum(slvr.chi_sqrd_result_gpu).get()

        if not self.weight_vector:
            slvr.set_X2(gpu_sum/slvr.sigma_sqrd)
        else:
            slvr.set_X2(gpu_sum)

    def post_execution(self, solver, stream=None):
        pass
