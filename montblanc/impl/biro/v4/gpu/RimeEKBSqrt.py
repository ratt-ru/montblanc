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
from pycuda.compiler import SourceModule

import montblanc
import montblanc.util as mbu
from montblanc.node import Node

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 8,    # Number of antennas
    'BLOCKDIMZ' : 2,    # Number of timesteps
    'maxregs'   : 32    # Maximum number of registers
}

# 44 registers results in some spillage into
# local memory
DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels
    'BLOCKDIMY' : 4,    # Number of antennas
    'BLOCKDIMZ' : 1,    # Number of timesteps
    'maxregs'   : 48    # Maximum number of registers
}

KERNEL_TEMPLATE = string.Template("""
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>
#include <montblanc/include/jones.cuh>

#define NA (${na})
#define NBL (${nbl})
#define NCHAN (${nchan})
#define NTIME (${ntime})
#define NSRC (${nsrc})
#define NPOL (4)
#define NPOLCHAN (NPOL*NCHAN)

#define BLOCKDIMX (${BLOCKDIMX})
#define BLOCKDIMY (${BLOCKDIMY})
#define BLOCKDIMZ (${BLOCKDIMZ})

#define TWO_PI_OVER_C (${two_pi_over_c})

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

template <typename T>
class EKBTraits {};

template <>
class EKBTraits<float> {
public:
    typedef float3 UVWType;
    typedef float2 LMType;
};

template <>
class EKBTraits<double> {
public:
    typedef double3 UVWType;
    typedef double2 LMType;
};

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_EKBSqrt_impl(
    typename EKBTraits<T>::UVWType * uvw,
    typename EKBTraits<T>::LMType * lm,
    T * frequency,
    typename Tr::ct * B_sqrt,
    typename Tr::ct * jones)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)

    if(TIME >= DEXT(ntime) || ANT >= DEXT(na) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ typename EKBTraits<T>::UVWType s_uvw[BLOCKDIMZ][BLOCKDIMY];

    // Shared Memory produces a faster kernel than
    // registers for some reason!
    __shared__ typename EKBTraits<T>::LMType s_lm;

    __shared__ T freq[BLOCKDIMX];

    int i;

    // UVW coordinates vary by antenna and time, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*NA + ANT;
        s_uvw[threadIdx.z][threadIdx.y] = uvw[i];
    }

    // Wavelengths vary by channel, not by time and antenna
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { freq[threadIdx.x] = frequency[POLCHAN>>2]; }

    __syncthreads();

    for(int SRC=0; SRC < DEXT(nsrc); ++SRC)
    {
        // LM coordinates vary only by source, not antenna, time or channel
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
        {
            i = SRC;   s_lm = lm[i];
        }
        __syncthreads();

        // Calculate the n coordinate.
        T n = Po::sqrt(T(1.0) - s_lm.x*s_lm.x - s_lm.y*s_lm.y) - T(1.0);

        // Calculate the phase term for this antenna
        T phase = s_uvw[threadIdx.z][threadIdx.y].z*n
            + s_uvw[threadIdx.z][threadIdx.y].y*s_lm.y
            + s_uvw[threadIdx.z][threadIdx.y].x*s_lm.x;

        phase *= T(-TWO_PI_OVER_C) * freq[threadIdx.x];

        typename Tr::ct cplx_phase;
        Po::sincos(phase, &cplx_phase.y, &cplx_phase.x);

        i = (SRC*NTIME + TIME)*NPOLCHAN + POLCHAN;
        // Load in the brightness square root
        typename Tr::ct brightness_sqrt = B_sqrt[i];
        montblanc::complex_multiply_in_place<T>(cplx_phase, brightness_sqrt);

        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        // Load in the E Beam, and multiply it by KB
        typename Tr::ct J = jones[i];
        montblanc::jones_multiply_4x4_in_place<T>(J, cplx_phase);

        // Write out the jones matrices
        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        jones[i] = J;
        __syncthreads();
    }
}

extern "C" {

#define stamp_rime_EKBSqrt_fn(ft,ct,uvw_type,lm_type) \
__global__ void \
rime_jones_EKBSqrt_ ## ft( \
    uvw_type * UVW, \
    lm_type * LM, \
    ft * frequency, \
    ct * B_sqrt, \
    ct * jones) \
{ \
    rime_jones_EKBSqrt_impl<ft>(UVW, LM, \
        frequency, B_sqrt, jones); \
}

stamp_rime_EKBSqrt_fn(float,float2,float3,float2)
stamp_rime_EKBSqrt_fn(double,double2,double3,double2)

} // extern "C" {
""")

class RimeEKBSqrt(Node):
    def __init__(self):
        super(RimeEKBSqrt, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver
        ntime, na, npolchan = slvr.dim_local_size('ntime', 'na', 'npolchan')

        # Get a property dictionary off the solver
        D = slvr.template_dict()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)
        D['rime_const_data_struct'] = slvr.const_data().string_def()

        D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'] = \
            mbu.redistribute_threads(
                D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'],
                npolchan, na, ntime)

        regs = str(FLOAT_PARAMS['maxregs'] \
                if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_jones_EKBSqrt_float' \
            if slvr.is_float() is True else \
            'rime_jones_EKBSqrt_double'

        kernel_string = KERNEL_TEMPLATE.substitute(**D)
        self.mod = SourceModule(kernel_string,
            options=['-lineinfo','-maxrregcount', regs],
            include_dirs=[montblanc.get_source_path()],
            no_extern_c=True)

        self.rime_const_data_gpu = self.mod.get_global('C')
        self.kernel = self.mod.get_function(kname)
        self.launch_params = self.get_launch_params(slvr, D)

    def shutdown(self, solver, stream=None):
        pass

    def pre_execution(self, solver, stream=None):
        pass

    def get_launch_params(self, slvr, D):
        polchans_per_block = D['BLOCKDIMX']
        ants_per_block = D['BLOCKDIMY']
        times_per_block = D['BLOCKDIMZ']

        ntime, na, npolchan = slvr.dim_local_size('ntime', 'na', 'npolchan')
        polchan_blocks = mbu.blocks_required(npolchan, polchans_per_block)
        ant_blocks = mbu.blocks_required(na, ants_per_block)
        time_blocks = mbu.blocks_required(ntime, times_per_block)

        return {
            'block' : (polchans_per_block, ants_per_block, times_per_block),
            'grid'  : (polchan_blocks, ant_blocks, time_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        if stream is not None:
            cuda.memcpy_htod_async(
                self.rime_const_data_gpu[0],
                slvr.const_data().ndary(),
                stream=stream)
        else:
            cuda.memcpy_htod(
                self.rime_const_data_gpu[0],
                slvr.const_data().ndary())

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.frequency_gpu,
            slvr.B_sqrt_gpu, slvr.jones_gpu,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
