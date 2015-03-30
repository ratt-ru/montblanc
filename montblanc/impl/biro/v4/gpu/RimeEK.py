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

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_EK_impl(
    T * uvw,
    T * lm,
    T * stokes,
    T * alpha,
    T * wavelength,
    T * point_errors,
    typename Tr::ct * jones,
    T ref_wave,
    T beam_width,
    T beam_clip)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)

    if(TIME >= NTIME || ANT >= NA || POLCHAN >= NPOLCHAN)
        return;

    __shared__ T u[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T v[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T w[BLOCKDIMZ][BLOCKDIMY];

    // Shared Memory produces a faster kernel than
    // registers for some reason!
    __shared__ T l;
    __shared__ T m;

    __shared__ T ld[BLOCKDIMZ][BLOCKDIMY];
    __shared__ T md[BLOCKDIMZ][BLOCKDIMY];

    __shared__ T wl[BLOCKDIMX];

    __shared__ T a[BLOCKDIMZ];

    int i;

    // UVW coordinates vary by antenna and time, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*NA + ANT;
        u[threadIdx.z][threadIdx.y] = uvw[i];
        ld[threadIdx.z][threadIdx.y] = point_errors[i];
        i += NA*NTIME;
        v[threadIdx.z][threadIdx.y] = uvw[i];
        md[threadIdx.z][threadIdx.y] = point_errors[i];
        i += NA*NTIME;
        w[threadIdx.z][threadIdx.y] = uvw[i];
    }

    // Wavelengths vary by channel, not by time and antenna
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { wl[threadIdx.x] = wavelength[POLCHAN>>2]; }

    __syncthreads();

    for(int SRC=0;SRC<NSRC;++SRC)
    {
        // LM coordinates vary only by source, not antenna, time or channel
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0)
        {
            i = SRC;   l = lm[i];
            i += NSRC; m = lm[i];
        }
        __syncthreads();

        // Calculate the n coordinate.
        T n = Po::sqrt(T(1.0) - l*l - m*m) - T(1.0);

        // Calculate the phase term for this antenna
        T phase = w[threadIdx.z][threadIdx.y]*n
            + v[threadIdx.z][threadIdx.y]*m
            + u[threadIdx.z][threadIdx.y]*l;

        phase *= T(2.0) * Tr::cuda_pi / wl[threadIdx.x];

        typename Tr::ct cplx_phase;
        Po::sincos(phase, &cplx_phase.y, &cplx_phase.x);

        i = (SRC*NTIME + TIME)*NPOL + POL;
        typename Tr::ft pol = stokes[i];
        typename Tr::ct brightness;

        montblanc::create_brightness<T>(brightness, pol);
        montblanc::complex_multiply_in_place<T>(cplx_phase, brightness);

        // Write out the phase terms
        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        jones[i] = cplx_phase;
        __syncthreads();
    }
}

extern "C" {

#define stamp_rime_ek_fn(ft,ct) \
__global__ void \
rime_jones_EK_ ## ft( \
    ft * UVW, \
    ft * LM, \
    ft * stokes, \
    ft * alpha, \
    ft * wavelength, \
    ft * point_errors, \
    ct * jones, \
    ft ref_wave, \
    ft beam_width, \
    ft beam_clip) \
{ \
    rime_jones_EK_impl<ft>(UVW, LM, stokes, alpha, wavelength, \
        point_errors, jones, ref_wave, beam_width, beam_clip); \
}

stamp_rime_ek_fn(float,float2)
stamp_rime_ek_fn(double,double2)

} // extern "C" {
""")

class RimeEK(Node):
    def __init__(self):
        super(RimeEK, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        self.pol_chans = 4*slvr.nchan

        # Get a property dictionary off the solver
        D = slvr.get_properties()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        # Update kernel parameters to cater for radically
        # smaller problem sizes. Caters for a subtle bug
        # with Kepler shuffles and warp sizes < 32
        if self.pol_chans < D['BLOCKDIMX']:
            D['BLOCKDIMX'] = self.pol_chans
        if slvr.na < D['BLOCKDIMY']:
            D['BLOCKDIMY'] = slvr.na
        if slvr.ntime < D['BLOCKDIMZ']:
            D['BLOCKDIMZ'] = slvr.ntime

        regs = str(FLOAT_PARAMS['maxregs'] \
                if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_jones_EK_float' \
            if slvr.is_float() is True else \
            'rime_jones_EK_double'

        kernel_string = KERNEL_TEMPLATE.substitute(**D)
        self.mod = SourceModule(kernel_string,
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
        pol_chans_per_block = D['BLOCKDIMX']
        ants_per_block = D['BLOCKDIMY']
        times_per_block = D['BLOCKDIMZ']

        pol_chan_blocks = self.blocks_required(self.pol_chans, pol_chans_per_block)
        ant_blocks = self.blocks_required(slvr.na, ants_per_block)
        time_blocks = self.blocks_required(slvr.ntime, times_per_block)

        return {
            'block' : (pol_chans_per_block, ants_per_block, times_per_block),
            'grid'  : (pol_chan_blocks, ant_blocks, time_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.stokes_gpu, slvr.alpha_gpu,
            slvr.wavelength_gpu, slvr.point_errors_gpu, slvr.jones_gpu,
            slvr.ref_wave, slvr.beam_width, slvr.beam_clip,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
