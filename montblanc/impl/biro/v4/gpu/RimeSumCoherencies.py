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
#include <montblanc/include/jones.cuh>

#define NA ${na}
#define NBL ${nbl}
#define NCHAN ${nchan}
#define NTIME ${ntime}
#define NPSRC ${npsrc}
#define NGSRC ${ngsrc}
#define NSSRC ${nssrc}
#define NSRC ${nsrc}
#define NPOL (4)
#define NPOLCHAN (NPOL*NCHAN)

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
void rime_sum_coherencies_impl(
    typename Tr::ft * uvw,
    typename Tr::ft * gauss_shape,
    typename Tr::ft * sersic_shape,
    typename Tr::ft * wavelength,
    int * ant_pairs,
    typename Tr::ct * jones,
    typename Tr::ft * weight_vector,
    typename Tr::ct * bayes_data,
    typename Tr::ct * G_term,
    typename Tr::ct * visibilities,
    typename Tr::ft * chi_sqrd_result)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    if(TIME >= NTIME || BL >= NBL || POLCHAN >= NPOLCHAN)
        return;

    __shared__ T u[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T v[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T w[BLOCKDIMZ][BLOCKDIMY];

    __shared__ T el;
    __shared__ T em;
    __shared__ T eR;

    __shared__ T e1;
    __shared__ T e2;
    __shared__ T sersic_scale;

    __shared__ T wl[BLOCKDIMX];

    int i;

    // Figure out the antenna pairs
    i = TIME*NBL + BL;   int ANT1 = ant_pairs[i];
    i += NBL*NTIME;      int ANT2 = ant_pairs[i];

    // UVW coordinates vary by baseline and time, but not polarised channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = TIME*NA + ANT1;    u[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         v[threadIdx.z][threadIdx.y] = uvw[i];
        i += NA*NTIME;         w[threadIdx.z][threadIdx.y] = uvw[i];

        i = TIME*NA + ANT2;    u[threadIdx.z][threadIdx.y] -= uvw[i];
        i += NA*NTIME;         v[threadIdx.z][threadIdx.y] -= uvw[i];
        i += NA*NTIME;         w[threadIdx.z][threadIdx.y] -= uvw[i];
    }

    // Wavelength varies by channel, but not baseline and time
    // TODO uses 4 times the actually required space, since
    // we don't need to store a wavelength per polarisation
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { wl[threadIdx.x] = wavelength[POLCHAN >> 2]; }

    // Complex Number containing the sum
    // for this polarisation
    typename Tr::ct polsum = Po::make_ct(0.0, 0.0);

    // Point Sources
    for(int SRC=0;SRC<NPSRC;++SRC)
    {
        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_one = jones[i];
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_two = jones[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_two, ant_one);

        polsum.x += ant_two.x;
        polsum.y += ant_two.y;
    }

    // Gaussian sources
    for(int SRC=NPSRC;SRC<NPSRC+NGSRC;++SRC)
    {
        // gaussian shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC-NPSRC;  el = gauss_shape[i];
            i += NGSRC;     em = gauss_shape[i];
            i += NGSRC;     eR = gauss_shape[i];
        }

        __syncthreads();

        T u1 = u[threadIdx.z][threadIdx.y]*em - v[threadIdx.z][threadIdx.y]*el;
        u1 *= T(GAUSS_SCALE)/wl[threadIdx.x];
        u1 *= eR;
        T v1 = u[threadIdx.z][threadIdx.y]*el + v[threadIdx.z][threadIdx.y]*em;
        v1 *= T(GAUSS_SCALE)/wl[threadIdx.x];
        T exp = Po::exp(-(u1*u1 +v1*v1));

        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_one = jones[i];
        // Multiple in the gaussian shape
        ant_one.x *= exp;
        ant_one.y *= exp;
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_two = jones[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_two, ant_one);

        polsum.x += ant_two.x;
        polsum.y += ant_two.y;

        __syncthreads();
    }

    // Sersic Sources
    for(int SRC=NPSRC+NGSRC;SRC<NPSRC+NGSRC+NSSRC;++SRC)
    {
        // sersic shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC-NPSRC-NGSRC;  e1 = sersic_shape[i];
            i += NSSRC;           e2 = sersic_shape[i];
            i += NSSRC;           sersic_scale = sersic_shape[i];
        }

        __syncthreads();

        // sersic source in  the Fourier domain
        T u1 = u[threadIdx.z][threadIdx.y]*(T(1.0)+e1) + v[threadIdx.z][threadIdx.y]*e2;
        u1 *= T(TWO_PI)/wl[threadIdx.x];
        u1 *= sersic_scale/(T(1.0)-e1*e1-e2*e2);
        T v1 = u[threadIdx.z][threadIdx.y]*e2 + v[threadIdx.z][threadIdx.y]*(T(1.0)-e1);
        v1 *= T(TWO_PI)/wl[threadIdx.x];
        v1 *= sersic_scale/(T(1.0)-e1*e1-e2*e2);
        T sersic_factor = T(1.0) + u1*u1+v1*v1;
        sersic_factor = T(1.0) / (sersic_factor*Po::sqrt(sersic_factor));


        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_one = jones[i];
        ant_one.x *= sersic_factor;
        ant_one.y *= sersic_factor;
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_two = jones[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_two, ant_one);

        polsum.x += ant_two.x;
        polsum.y += ant_two.y;
    }

    // Multiply the visibility by antenna 2's g term
    i = (TIME*NA + ANT2)*NPOLCHAN + POLCHAN;
    typename Tr::ct ant2_g_term = G_term[i];
    montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant2_g_term, polsum);

    // Multiply the visibility by antenna 1's g term
    i = (TIME*NA + ANT1)*NPOLCHAN + POLCHAN;
    typename Tr::ct ant1_g_term = G_term[i];
    montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant2_g_term, ant1_g_term);

    // Write out the visibilities
    i = (TIME*NBL + BL)*NPOLCHAN + POLCHAN;
    visibilities[i] = ant2_g_term;

    // Compute the chi squared sum terms
    typename Tr::ct delta = bayes_data[i];
    delta.x -= ant2_g_term.x; delta.y -= ant2_g_term.y;
    delta.x *= delta.x; delta.y *= delta.y;

    // Apply any necessary weighting factors
    if(apply_weights)
    {
        T w = weight_vector[i];
        delta.x *= w;
        delta.y *= w;
    }

    // Now, add the real and imaginary components
    // of each adjacent group of four polarisations
    // into the first polarisation.
    typename Tr::ct other = cub::ShuffleIndex(delta, cub::LaneId() + 2);

    // And polarisations 2 and 3 to 0 and 1
    if((POLCHAN & 0x3) < 2)
    {
        delta.x += other.x;
        delta.y += other.y;
    }

    other = cub::ShuffleIndex(delta, cub::LaneId() + 1);

    // And polarisation 1 to 0
    // and write out this chi squared sum term
    // if this is the first polarisation
    if((POLCHAN & 0x3) == 0) {
        delta.x += other.x;
        delta.y += other.y;

        i = (TIME*NBL + BL)*NCHAN + (POLCHAN >> 2);
        chi_sqrd_result[i] = delta.x + delta.y;
    }
}

extern "C" {

// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - apply_weights: boolean indicating whether we're weighting our visibilities
// - symbol: u or w depending on whether we're handling unweighted/weighted visibilities.

#define stamp_sum_coherencies_fn(ft, ct, apply_weights, symbol) \
__global__ void \
rime_sum_coherencies_ ## symbol ## chi_ ## ft( \
    ft * uvw, \
    ft * gauss_shape, \
    ft * sersic_shape, \
    ft * wavelength, \
    int * ant_pairs, \
    ct * jones, \
    ft * weight_vector, \
    ct * bayes_data, \
    ct * G_term, \
    ct * visibilities, \
    ft * chi_sqrd_result) \
{ \
    rime_sum_coherencies_impl<ft, apply_weights>(uvw, gauss_shape, sersic_shape, \
        wavelength, ant_pairs, jones, \
        weight_vector, bayes_data, G_term, \
        visibilities, chi_sqrd_result); \
}

stamp_sum_coherencies_fn(float, float2, false, u)
stamp_sum_coherencies_fn(float, float2, true, w)
stamp_sum_coherencies_fn(double, double2, false, u)
stamp_sum_coherencies_fn(double, double2, true, w)

} // extern "C" {
""")

class RimeSumCoherencies(Node):
    def __init__(self, weight_vector=False):
        super(RimeSumCoherencies, self).__init__()
        self.weight_vector = weight_vector

    def initialise(self, solver, stream=None):
        slvr = solver

        self.polchans = 4*slvr.nchan

        D = slvr.get_properties()
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        # Update kernel parameters to cater for radically
        # smaller problem sizes. Caters for a subtle bug
        # with Kepler shuffles and warp sizes < 32
        if self.polchans < D['BLOCKDIMX']: D['BLOCKDIMX'] = self.polchans
        if slvr.nbl < D['BLOCKDIMY']: D['BLOCKDIMY'] = slvr.nbl
        if slvr.ntime < D['BLOCKDIMZ']: D['BLOCKDIMZ'] = slvr.ntime

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_sum_coherencies_' + \
            ('w' if self.weight_vector else 'u') + 'chi_' + \
            ('float' if slvr.is_float() is True else 'double')

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        self.kernel = self.mod.get_function(kname)
        self.launch_params = self.get_launch_params(slvr, D)

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def get_launch_params(self, slvr, D):
        polchans_per_block = D['BLOCKDIMX']
        bl_per_block = D['BLOCKDIMY']
        times_per_block = D['BLOCKDIMZ']

        polchan_blocks = self.blocks_required(self.polchans, polchans_per_block)
        bl_blocks = self.blocks_required(slvr.nbl, bl_per_block)
        time_blocks = self.blocks_required(slvr.ntime, times_per_block)

        return {
            'block' : (polchans_per_block, bl_per_block, times_per_block),
            'grid'  : (polchan_blocks, bl_blocks, time_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape_gpu

        sersic = np.intp(0) if np.product(slvr.sersic_shape_shape) == 0 \
            else slvr.sersic_shape_gpu

        self.kernel(slvr.uvw_gpu, gauss, sersic,
            slvr.wavelength_gpu, slvr.ant_pairs_gpu,
            slvr.jones_gpu, slvr.weight_vector_gpu,
            slvr.bayes_data_gpu, slvr.G_term_gpu,
            slvr.vis_gpu, slvr.chi_sqrd_result_gpu,
            **self.launch_params)

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