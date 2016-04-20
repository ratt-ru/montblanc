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

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import montblanc
import montblanc.util as mbu
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX': 32,    # Number of channels*4 polarisations
    'BLOCKDIMY': 8,     # Number of baselines
    'BLOCKDIMZ': 1,     # Number of timesteps
    'maxregs': 48       # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX': 32,    # Number of channels*4 polarisations
    'BLOCKDIMY': 8,     # Number of baselines
    'BLOCKDIMZ': 1,     # Number of timesteps
    'maxregs': 63       # Maximum number of registers
}

KERNEL_TEMPLATE = string.Template("""
#include <stdint.h>
#include <cstdio>
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>
#include <montblanc/include/jones.cuh>

#define BLOCKDIMX ${BLOCKDIMX}
#define BLOCKDIMY ${BLOCKDIMY}
#define BLOCKDIMZ ${BLOCKDIMZ}

#define GAUSS_SCALE ${gauss_scale}
#define TWO_PI_OVER_C ${two_pi_over_c}

template <typename T>
class SumCohTraits {};

template <>
class SumCohTraits<float> {
public:
    typedef float3 UVWType;
};

template <>
class SumCohTraits<double> {
public:
    typedef double3 UVWType;
};

// Here, the definition of the
// rime_const_data struct
// is inserted into the template
// An area of constant memory
// containing an instance of this
// structure is declared. 
${rime_const_data_struct}
__constant__ rime_const_data C;
#define LEXT(name) C.name.extents[0]
#define UEXT(name) C.name.extents[1]
#define DEXT(name) (C.name.extents[1] - C.name.extents[0])
#define GLOBAL(name) (C.name.global_size)

#define NA (C.na.local_size)
#define NBL (C.nbl.local_size)
#define NCHAN (C.nchan.local_size)
#define NTIME (C.ntime.local_size)
#define NPSRC (C.npsrc.local_size)
#define NGSRC (C.ngsrc.local_size)
#define NSSRC (C.nssrc.local_size)
#define NSRC (C.nnsrc.local_size)
#define NPOL (C.npol.local_size)
#define NPOLCHAN (C.npolchan.local_size)

template <
    typename T,
    bool apply_weights=false,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_sum_coherencies_impl(
    typename SumCohTraits<T>::UVWType * uvw,
    typename Tr::ft * gauss_shape,
    typename Tr::ft * sersic_shape,
    typename Tr::ft * frequency,
    int * ant_pairs,
    typename Tr::ct * jones,
    uint8_t * flag,
    typename Tr::ft * weight_vector,
    typename Tr::ct * observed_vis,
    typename Tr::ct * G_term,
    typename Tr::ct * visibilities,
    typename Tr::ft * chi_sqrd_result)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    if(TIME >= DEXT(ntime) || BL >= DEXT(nbl) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ struct {
        typename SumCohTraits<T>::UVWType uvw[BLOCKDIMZ][BLOCKDIMY];

        T el;
        T em;
        T eR;

        T e1;
        T e2;
        T sersic_scale;

        T freq[BLOCKDIMX];

    } shared;

    T & U = shared.uvw[threadIdx.z][threadIdx.y].x; 
    T & V = shared.uvw[threadIdx.z][threadIdx.y].y; 
    T & W = shared.uvw[threadIdx.z][threadIdx.y].z; 


    int i;

    // Figure out the antenna pairs
    i = TIME*NBL + BL;   int ANT1 = ant_pairs[i];
    i += NBL*NTIME;      int ANT2 = ant_pairs[i];

    // UVW coordinates vary by baseline and time, but not polarised channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = TIME*NA + ANT1;
        shared.uvw[threadIdx.z][threadIdx.y] = uvw[i];

        i = TIME*NA + ANT2;
        typename SumCohTraits<T>::UVWType ant2_uvw = uvw[i];
        U -= ant2_uvw.x;
        V -= ant2_uvw.y;
        W -= ant2_uvw.z;
    }

    // Wavelength varies by channel, but not baseline and time
    // TODO uses 4 times the actually required space, since
    // we don't need to store a frequency per polarisation
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { shared.freq[threadIdx.x] = frequency[POLCHAN >> 2]; }

    // We process sources in batches, accumulating visibility values.
    // Visibilities are read in and written out at the start and end
    // of each batch respectively.

    // Initialise polarisation to zero if this is the first batch.
    // Otherwise, read in the visibilities from the previous batch.
    typename Tr::ct polsum = Po::make_ct(0.0, 0.0);
    if(LEXT(nsrc) > 0)
    {
        i = (TIME*NBL + BL)*NPOLCHAN + POLCHAN;
        polsum = visibilities[i];
    }

    int SRC_START = 0;
    int SRC_STOP = DEXT(npsrc);

    // Point Sources
    for(int SRC = SRC_START; SRC < SRC_STOP; ++SRC)
    {
        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_one = jones[i];
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_two = jones[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_one, ant_two);

        polsum.x += ant_one.x;
        polsum.y += ant_one.y;
    }

    SRC_START = SRC_STOP;
    SRC_STOP += DEXT(ngsrc);

    // Gaussian sources
    for(int SRC = SRC_START; SRC < SRC_STOP; ++SRC)
    {
        // gaussian shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC - SRC_START;  shared.el = gauss_shape[i];
            i += DEXT(ngsrc);       shared.em = gauss_shape[i];
            i += DEXT(ngsrc);       shared.eR = gauss_shape[i];
        }

        __syncthreads();

        // Create references to a
        // complicated part of shared memory
        const T & U = shared.uvw[threadIdx.z][threadIdx.y].x; 
        const T & V = shared.uvw[threadIdx.z][threadIdx.y].y; 

        T u1 = U*shared.em - V*shared.el;
        u1 *= T(GAUSS_SCALE)*shared.freq[threadIdx.x];
        u1 *= shared.eR;
        T v1 = U*shared.el + V*shared.em;
        v1 *= T(GAUSS_SCALE)*shared.freq[threadIdx.x];
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
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_one, ant_two);

        polsum.x += ant_one.x;
        polsum.y += ant_one.y;

        __syncthreads();
    }

    SRC_START = SRC_STOP;
    SRC_STOP += DEXT(nssrc);

    // Sersic Sources
    for(int SRC = SRC_START; SRC < SRC_STOP; ++SRC)
    {
        // sersic shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC - SRC_START;                shared.e1 = sersic_shape[i];
            i += DEXT(nssrc);                   shared.e2 = sersic_shape[i];
            i += DEXT(nssrc);                   shared.sersic_scale = sersic_shape[i];
        }

        __syncthreads();

        // Create references to a
        // complicated part of shared memory
        const T & U = shared.uvw[threadIdx.z][threadIdx.y].x; 
        const T & V = shared.uvw[threadIdx.z][threadIdx.y].y; 

        // sersic source in  the Fourier domain
        T u1 = U*(T(1.0)+shared.e1) + V*shared.e2;
        u1 *= T(TWO_PI_OVER_C)*shared.freq[threadIdx.x];
        u1 *= shared.sersic_scale/(T(1.0)-shared.e1*shared.e1-shared.e2*shared.e2);
        T v1 = U*shared.e2 + V*(T(1.0)-shared.e1);
        v1 *= T(TWO_PI_OVER_C)*shared.freq[threadIdx.x];
        v1 *= shared.sersic_scale/(T(1.0)-shared.e1*shared.e1-shared.e2*shared.e2);
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
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_one, ant_two);

        polsum.x += ant_one.x;
        polsum.y += ant_one.y;
    }

    // If the upper extent of this source batch is less than
    // the global number of sources, then we've still got more
    // batches to evaluate. Write visibilities out to global memory
    // and return
    if(UEXT(nsrc) < GLOBAL(nsrc))
    {
        i = (TIME*NBL + BL)*NPOLCHAN + POLCHAN;
        visibilities[i] = polsum;
        return;
    }

    // Otherwise, apply G terms and calculate chi-squared terms

    // Multiply the visibility by antenna 1's g term
    i = (TIME*NA + ANT1)*NPOLCHAN + POLCHAN;
    typename Tr::ct ant1_g_term = G_term[i];
    montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant1_g_term, polsum);

    // Multiply the visibility by antenna 2's g term
    i = (TIME*NA + ANT2)*NPOLCHAN + POLCHAN;
    typename Tr::ct ant2_g_term = G_term[i];
    montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant1_g_term, ant2_g_term);

    // Compute the chi squared sum terms
    i = (TIME*NBL + BL)*NPOLCHAN + POLCHAN;
    typename Tr::ct delta = observed_vis[i];

    // Zero the polarisation if it is flagged
    if(flag[i] > 0)
    {
        ant1_g_term.x = 0; ant1_g_term.y = 0;
        delta.x = 0; delta.y = 0;
    }

    // Write out the visibilities
    visibilities[i] = ant1_g_term;

    delta.x -= ant1_g_term.x; delta.y -= ant1_g_term.y;
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

    // Add polarisations 2 and 3 to 0 and 1
    if((POLCHAN & 0x3) < 2)
    {
        delta.x += other.x;
        delta.y += other.y;
    }

    other = cub::ShuffleIndex(delta, cub::LaneId() + 1);

    // If this is the polarisation 0, add polarisation 1
    // and write out this chi squared sum term
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

#define stamp_sum_coherencies_fn(ft, ct, uvwt, apply_weights, symbol) \
__global__ void \
rime_sum_coherencies_ ## symbol ## chi_ ## ft( \
    uvwt * uvw, \
    ft * gauss_shape, \
    ft * sersic_shape, \
    ft * frequency, \
    int * ant_pairs, \
    ct * jones, \
    uint8_t * flag, \
    ft * weight_vector, \
    ct * observed_vis, \
    ct * G_term, \
    ct * visibilities, \
    ft * chi_sqrd_result) \
{ \
    rime_sum_coherencies_impl<ft, apply_weights>(uvw, gauss_shape, sersic_shape, \
        frequency, ant_pairs, jones, flag, \
        weight_vector, observed_vis, G_term, \
        visibilities, chi_sqrd_result); \
}

stamp_sum_coherencies_fn(float, float2, float3, false, u)
stamp_sum_coherencies_fn(float, float2, float3, true, w)
stamp_sum_coherencies_fn(double, double2, double3, false, u)
stamp_sum_coherencies_fn(double, double2, double3, true, w)

} // extern "C" {
""")

class RimeSumCoherencies(Node):
    def __init__(self):
        super(RimeSumCoherencies, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver
        ntime, nbl, npolchan = slvr.dim_local_size('ntime', 'nbl', 'npolchan')

        # Get a property dictionary off the solver
        D = slvr.template_dict()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)
        D['rime_const_data_struct'] = slvr.const_data().string_def()

        D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'] = \
            mbu.redistribute_threads(
                D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'],
                npolchan, nbl, ntime)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_sum_coherencies_' + \
            ('w' if slvr.use_weight_vector() else 'u') + 'chi_' + \
            ('float' if slvr.is_float() is True else 'double')

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        self.rime_const_data = self.mod.get_global('C')
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

        ntime, nbl, npolchan = slvr.dim_local_size('ntime', 'nbl', 'npolchan')
        polchan_blocks = mbu.blocks_required(npolchan, polchans_per_block)
        bl_blocks = mbu.blocks_required(nbl, bl_per_block)
        time_blocks = mbu.blocks_required(ntime, times_per_block)

        return {
            'block' : (polchans_per_block, bl_per_block, times_per_block),
            'grid'  : (polchan_blocks, bl_blocks, time_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        if stream is not None:
            cuda.memcpy_htod_async(
                self.rime_const_data[0],
                slvr.const_data().ndary(),
                stream=stream)
        else:
            cuda.memcpy_htod(
                self.rime_const_data[0],
                slvr.const_data().ndary())

        # The gaussian shape array can be empty if
        # no gaussian sources were specified.
        gauss = np.intp(0) if np.product(slvr.gauss_shape_shape) == 0 \
            else slvr.gauss_shape

        sersic = np.intp(0) if np.product(slvr.sersic_shape_shape) == 0 \
            else slvr.sersic_shape

        self.kernel(slvr.uvw, gauss, sersic,
            slvr.frequency, slvr.ant_pairs,
            slvr.jones, slvr.flag, slvr.weight_vector,
            slvr.observed_vis, slvr.G_term,
            slvr.model_vis, slvr.chi_sqrd_result,
            stream=stream, **self.launch_params)

        # Call the pycuda reduction kernel.
        # Divide by the single sigma squared value if a weight vector
        # is not required. Otherwise the kernel will incorporate the
        # individual sigma squared values into the sum
        gpu_sum = gpuarray.sum(slvr.chi_sqrd_result).get()

        if not slvr.use_weight_vector():
            slvr.set_X2(gpu_sum/slvr.sigma_sqrd)
        else:
            slvr.set_X2(gpu_sum)

    def post_execution(self, solver, stream=None):
        pass
