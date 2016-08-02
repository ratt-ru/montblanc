#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Marzia Rivi, Simon Perkins
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


from montblanc.config import RimeSolverConfig as Options

FLOAT_PARAMS = {
    'BLOCKDIMX': 4,    # Number of channels*4 polarisations
    'BLOCKDIMY': 25,     # Number of baselines
    'BLOCKDIMZ': 10,     # Number of timesteps
    'maxregs': 48       # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX': 4,    # Number of channels*4 polarisations
    'BLOCKDIMY': 25,     # Number of baselines
    'BLOCKDIMZ': 10,     # Number of timesteps
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
#define LEXT(name) C.name.lower_extent
#define UEXT(name) C.name.upper_extent
#define DEXT(name) (C.name.upper_extent - C.name.lower_extent)

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


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


template <
    typename T,
    bool apply_weights=false,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void sersic_chi_squared_gradient_impl(
    typename SumCohTraits<T>::UVWType * uvw,
    typename Tr::ft * sersic_shape,
    typename Tr::ft * frequency,
    int * antenna1,
    int * antenna2,
    typename Tr::ct * jones,
    uint8_t * flag,
    typename Tr::ft * weight_vector,
    typename Tr::ct * observed_vis,
    typename Tr::ct * G_term,
    typename Tr::ct * visibilities,
    typename Tr::ft * X2_grad)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    if(TIME >= DEXT(ntime) || BL >= DEXT(nbl) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ struct {
        typename SumCohTraits<T>::UVWType uvw[BLOCKDIMZ][BLOCKDIMY];

        T e1;
        T e2;
        T sersic_scale;

        T X2_grad_part[3];

        T freq[BLOCKDIMX];

    } shared;

    typename Tr::ct dev_vis[3];

    T & U = shared.uvw[threadIdx.z][threadIdx.y].x; 
    T & V = shared.uvw[threadIdx.z][threadIdx.y].y; 
    T & W = shared.uvw[threadIdx.z][threadIdx.y].z; 

    int i;

    // Figure out the antenna pairs
    i = TIME*NBL + BL;   
    int ANT1 = antenna1[i];
    int ANT2 = antenna2[i];

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

    i = (TIME*NBL +BL)* NPOLCHAN + POLCHAN;
    typename Tr::ct delta = observed_vis[i];
    delta.x -= visibilities[i].x;
    delta.y -= visibilities[i].y;

    // Zero the polarisation if it is flagged
    if(flag[i] > 0)
    {
        delta.x = 0; delta.y = 0;
    }

    // Apply any necessary weighting factors
    if(apply_weights)
    {
        T w = weight_vector[i];
        delta.x *= w;
        delta.y *= w;
    }

    int SRC_START = DEXT(npsrc) + DEXT(ngsrc);
    int SRC_STOP = SRC_START + DEXT(nssrc);

    // Loop over Sersic Sources
    for(int SRC = SRC_START; SRC < SRC_STOP; ++SRC)
    {
        // sersic shape only varies by source. Shape parameters
        // thus apply to the entire block and we can load them with
        // only the first thread.
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC - SRC_START;                 shared.e1 = sersic_shape[i];
            i += DEXT(nssrc);                    shared.e2 = sersic_shape[i];
            i += DEXT(nssrc);                    shared.sersic_scale = sersic_shape[i];
        }
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z < 3)
            shared.X2_grad_part[threadIdx.z] = 0.;

        __syncthreads();

        // Create references to a
        // complicated part of shared memory
        const T & U = shared.uvw[threadIdx.z][threadIdx.y].x; 
        const T & V = shared.uvw[threadIdx.z][threadIdx.y].y; 

        // sersic source in  the Fourier domain
        T u1 = U*(T(1.0)+shared.e1) + V*shared.e2;
        T v1 = U*shared.e2 + V*(T(1.0)-shared.e1);

        T sersic_factor = T(1.0)-shared.e1*shared.e1-shared.e2*shared.e2;
        T dev_e1 = u1*(U*sersic_factor + 2*shared.e1*u1);
        dev_e1 += v1*(2*shared.e1*v1 - V*sersic_factor);
        T dev_e2 = u1*(V*sersic_factor + 2*shared.e2*u1);
        dev_e2 += v1*(U*sersic_factor + 2*shared.e2*v1);

        // pay attention to the instructions order because temp and sersic_factor variables are utilised twice!!!
        T temp = T(TWO_PI_OVER_C)*shared.freq[threadIdx.x]*shared.sersic_scale;
        temp /= sersic_factor;
        temp *= temp;
        dev_e1 *= temp/sersic_factor;
        dev_e2 *= temp/sersic_factor;     
   
        temp *= (u1*u1+v1*v1);
        sersic_factor = T(1.0)+temp;
        dev_e1 /= sersic_factor;
        dev_e2 /= sersic_factor;

        T dev_scale = temp/(shared.sersic_scale*sersic_factor);

        // Get the complex scalars for antenna two and multiply
        // in the exponent term
        // Get the complex scalar for antenna one and conjugate it
        i = ((SRC*NTIME + TIME)*NA + ANT1)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_one = jones[i];
        sersic_factor = T(1.0) / (sersic_factor*Po::sqrt(sersic_factor));
        ant_one.x *= sersic_factor;
        ant_one.y *= sersic_factor;
        i = ((SRC*NTIME + TIME)*NA + ANT2)*NPOLCHAN + POLCHAN;
        typename Tr::ct ant_two = jones[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant_one, ant_two);

        dev_vis[0].x = ant_one.x*dev_e1;
        dev_vis[0].y = ant_one.y*dev_e1;
        dev_vis[1].x = ant_one.x*dev_e2;
        dev_vis[1].y = ant_one.y*dev_e2;
        dev_vis[2].x = ant_one.x*dev_scale;
        dev_vis[2].y = ant_one.y*dev_scale;
   /*
        for (int p = 0; p < 3; p++)
        {
          // Multiply the visibility derivative by antenna 1's g term
          i = (TIME*NA + ANT1)*NPOLCHAN + POLCHAN;
          typename Tr::ct ant1_g_term = G_term[i];
          montblanc::jones_multiply_4x4_in_place<T>(ant1_g_term, dev_vis[p]);

          // Multiply the visibility by antenna 2's g term
          i = (TIME*NA + ANT2)*NPOLCHAN + POLCHAN;
          typename Tr::ct ant2_g_term = G_term[i];
          montblanc::jones_multiply_4x4_hermitian_transpose_in_place<T>(ant1_g_term, ant2_g_term);    
          dev_vis[p] = ant1_g_term;
        }
   
    */
        // Write partial derivative with respect to sersic parameters
        for (int p=0; p<3; p++)
        {
          dev_vis[p].x *= delta.x;
          dev_vis[p].y *= delta.y;

          typename Tr::ct other = cub::ShuffleIndex(dev_vis[p], cub::LaneId() + 2);
          // Add polarisations 2 and 3 to 0 and 1
          if((POLCHAN & 0x3) < 2)
          {
            dev_vis[p].x += other.x;
            dev_vis[p].y += other.y;
          }
          other = cub::ShuffleIndex(dev_vis[p], cub::LaneId() + 1);

          // If this is the polarisation 0, add polarisation 1
          // and write out this chi squared grad term
          if((POLCHAN & 0x3) == 0) 
          {
            dev_vis[p].x += other.x;
            dev_vis[p].y += other.y;

            //atomic addition to avoid concurrent access in the shared memory
            dev_vis[p].x += dev_vis[p].y;
            dev_vis[p].x *= 6;
            atomicAdd(&(shared.X2_grad_part[p]), dev_vis[p].x);
          }
        }
        __syncthreads();
        
        //atomic addition to avoid concurrent access in the device memory (contribution for a single particle)
        // 3 different threads writes each a different component to avoid serialisation
        if (threadIdx.x < 3 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            i = SRC - DEXT(npsrc) - DEXT(ngsrc) + threadIdx.x*DEXT(nssrc);
            atomicAdd(&(X2_grad[i]), shared.X2_grad_part[threadIdx.x]);
        }
    }
}

extern "C" {

// Macro that stamps out different kernels, depending
// on whether we're handling floats or doubles
// Arguments
// - ft: The floating point type. Should be float/double.
// - ct: The complex type. Should be float2/double2.
// - apply_weights: boolean indicating whether we're weighting our visibilities

#define stamp_sersic_chi_squared_gradient_fn(ft, ct, uvwt, apply_weights) \
__global__ void \
sersic_chi_squared_gradient( \
    uvwt * uvw, \
    ft * sersic_shape, \
    ft * frequency, \
    int * antenna1, \
    int * antenna2, \
    ct * jones, \
    uint8_t * flag,\
    ft * weight_vector, \
    ct * observed_vis, \
    ct * G_term, \
    ct * visibilities, \
    ft * X2_grad) \
{ \
    sersic_chi_squared_gradient_impl<ft, apply_weights>(uvw, sersic_shape, \
        frequency, antenna1, antenna2, jones, flag,\
        weight_vector, observed_vis, G_term, \
        visibilities, X2_grad); \
}

${stamp_function}

} // extern "C" {
""")

class SersicChiSquaredGradient(Node):
    def __init__(self, weight_vector=False):
        super(SersicChiSquaredGradient, self).__init__()
        self.weight_vector = weight_vector

    def initialise(self, solver, stream=None):
        slvr = solver
        nssrc, ntime, nbl, npolchan = slvr.dim_local_size('nssrc','ntime', 'nbl', 'npolchan')

          # Get a property dictionary off the solver
        D = slvr.template_dict()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)
        D['rime_const_data_struct'] = slvr.const_data().string_def()

        D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'] = \
            mbu.redistribute_threads(D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'],
            npolchan, nbl, ntime)

        regs = str(FLOAT_PARAMS['maxregs'] \
            if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        # Create the signature of the call to the function stamping macro
        stamp_args = ', '.join([
            'float' if slvr.is_float() else 'double',
            'float2' if slvr.is_float() else 'double2',
            'float3' if slvr.is_float() else 'double3',
            'true' if slvr.use_weight_vector() else 'false'])
        stamp_fn = ''.join(['stamp_sersic_chi_squared_gradient_fn(', stamp_args, ')'])
        D['stamp_function'] = stamp_fn

        kname = 'sersic_chi_squared_gradient'

        self.mod = SourceModule(
            KERNEL_TEMPLATE.substitute(**D),
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        if slvr.is_float():
            self.gradient = gpuarray.zeros((3,nssrc,),dtype=np.float32)
        else:
            self.gradient = gpuarray.zeros((3,nssrc,),dtype=np.float64)

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

        sersic = np.intp(0) if np.product(slvr.sersic_shape.shape) == 0 \
            else slvr.sersic_shape

        self.gradient.fill(0.) 

        self.kernel(slvr.uvw, sersic,
            slvr.frequency, slvr.antenna1, slvr.antenna2,
            slvr.jones, slvr.flag, slvr.weight_vector,
            slvr.observed_vis, slvr.G_term,
            slvr.model_vis, self.gradient,
            stream=stream, **self.launch_params)

        slvr.X2_grad = self.gradient.get() 

        if not self.weight_vector:
            slvr.X2_grad = slvr.X2_grad/slvr.sigma_sqrd

    def post_execution(self, solver, stream=None):
        pass
