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
    'BLOCKDIMX' : 32,   # Number of channels and polarisations
    'BLOCKDIMY' : 32,    # Number of timesteps
    'BLOCKDIMZ' : 1,    #
    'maxregs'   : 32    # Maximum number of registers
}

# 44 registers results in some spillage into
# local memory
DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels and polarisations
    'BLOCKDIMY' : 16,    # Number of timesteps
    'BLOCKDIMZ' : 1,    #
    'maxregs'   : 48    # Maximum number of registers
}

KERNEL_TEMPLATE = string.Template("""
#include \"math_constants.h\"
#include <montblanc/include/abstraction.cuh>
#include <montblanc/include/brightness.cuh>

#define BLOCKDIMX (${BLOCKDIMX})
#define BLOCKDIMY (${BLOCKDIMY})
#define BLOCKDIMZ (${BLOCKDIMZ})

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

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_B_sqrt_impl(
    T * stokes,
    T * alpha,
    T * frequency,
    T * ref_frequency,
    typename Tr::ct * B_sqrt)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)

    if(SRC >= DEXT(nsrc) || TIME >= DEXT(ntime) || POLCHAN >= DEXT(npolchan))
        { return; }

    __shared__ T freq_ratio[BLOCKDIMX];

    // TODO. Using 3 times more shared memory than we
    // really require here, since there's only
    // one frequency per channel.
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        freq_ratio[threadIdx.x] = frequency[POLCHAN>>2] / 
            ref_frequency[POLCHAN>>2];
    }

    __syncthreads();

    // Calculate the power term
    int i = SRC*NTIME + TIME;
    typename Tr::ft power = Po::pow(freq_ratio[threadIdx.x], alpha[i]);

    // Read in the stokes parameter,
    // multiplying it by the power term
    i = i*NPOL + POL;
    typename Tr::ft pol = stokes[i]*power;
    typename Tr::ct B_square_root;

    // Create the square root of the brightness matrix
    montblanc::create_brightness_sqrt<T>(B_square_root, pol);

    // Write out the square root of the brightness
    i = (SRC*NTIME + TIME)*NPOLCHAN + POLCHAN;
    B_sqrt[i] = B_square_root;
}



// Case NPOL = 1

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones1_B_sqrt_impl(
    T * stokes,
    T * alpha,
    T * frequency,
    T * ref_frequency,
    typename Tr::ct * B_sqrt)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;

    if(SRC >= DEXT(nsrc) || TIME >= DEXT(ntime) || POLCHAN >= DEXT(nchan))
        { return; }

    __shared__ T freq_ratio[BLOCKDIMX];

    // TODO. Using 3 times more shared memory than we
    // really require here, since there's only
    // one frequency per channel.
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        freq_ratio[threadIdx.x] = frequency[POLCHAN] / 
            ref_frequency[POLCHAN];
    }

    __syncthreads();

    // Calculate the power term
    int i = SRC*NTIME + TIME;
    typename Tr::ft power = Po::pow(freq_ratio[threadIdx.x], alpha[i]);

    // Read in the stokes parameter,
    // multiplying it by the power term
    typename Tr::ft pol = stokes[i]*power;

    // Write out the square root of the brightness
    i = (SRC*NTIME + TIME)*NPOLCHAN + POLCHAN;
    B_sqrt[i].x = Po::sqrt(pol);
    B_sqrt[i].y = 0.;
}




extern "C" {

#define stamp_jones_B_sqrt_fn(ft,ct) \
__global__ void \
rime_jones_B_sqrt_ ## ft( \
    ft * stokes, \
    ft * alpha, \
    ft * frequency, \
    ft * ref_frequency, \
    ct * B_sqrt) \
{ \
    rime_jones_B_sqrt_impl<ft>(stokes, alpha, \
        frequency, ref_frequency, B_sqrt); \
}

#define stamp_jones1_B_sqrt_fn(ft,ct) \
__global__ void \
rime_jones1_B_sqrt_ ## ft( \
    ft * stokes, \
    ft * alpha, \
    ft * frequency, \
    ft * ref_frequency, \
    ct * B_sqrt) \
{ \
    rime_jones1_B_sqrt_impl<ft>(stokes, alpha, \
        frequency, ref_frequency, B_sqrt); \
}


stamp_jones_B_sqrt_fn(float,float2);
stamp_jones_B_sqrt_fn(double,double2);

stamp_jones1_B_sqrt_fn(float,float2);
stamp_jones1_B_sqrt_fn(double,double2);


} // extern "C" {
""")

class RimeBSqrt(Node):
    def __init__(self):
        super(RimeBSqrt, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver
        nsrc, ntime, npolchan, npol = slvr.dim_local_size(
            'nsrc', 'ntime', 'npolchan', 'npol')

        # Get a property dictionary off the solver
        D = slvr.template_dict()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)
        D['rime_const_data_struct'] = slvr.const_data().string_def()

        D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'] = \
            mbu.redistribute_threads(
                D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'],
                npolchan, ntime, nsrc)

        regs = str(FLOAT_PARAMS['maxregs'] \
                if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        if npol == 1:
            kname = 'rime_jones1_B_sqrt_float' \
                if slvr.is_float() is True else \
                'rime_jones1_B_sqrt_double'
        else:
            kname = 'rime_jones_B_sqrt_float' \
                if slvr.is_float() is True else \
                'rime_jones_B_sqrt_double'

        kernel_string = KERNEL_TEMPLATE.substitute(**D)

        self.mod = SourceModule(kernel_string,
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
        times_per_block = D['BLOCKDIMY']
        srcs_per_block = D['BLOCKDIMZ']

        nsrc, ntime, npolchan = slvr.dim_local_size('nsrc', 'ntime', 'npolchan')
        polchan_blocks = mbu.blocks_required(npolchan, polchans_per_block)
        time_blocks = mbu.blocks_required(ntime, times_per_block)
        src_blocks = mbu.blocks_required(nsrc, srcs_per_block)

        return {
            'block' : (polchans_per_block, times_per_block, srcs_per_block),
            'grid'  : (polchan_blocks, time_blocks, src_blocks),
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

        self.kernel(slvr.stokes, slvr.alpha,
            slvr.frequency, slvr.ref_frequency, slvr.B_sqrt,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
