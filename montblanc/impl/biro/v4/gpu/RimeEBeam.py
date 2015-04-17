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
void rime_jones_E_beam_impl(
    T * lm,
    T * wavelength,
    T * point_errors,
    typename Tr::ct * E_beam,
    T * beam_rot_vel)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)

    if(SRC >= NSRC || ANT >= NA || POLCHAN >= NPOLCHAN)
        return;

    __shared__ T wl[BLOCKDIMX];

    __shared__ T l0[BLOCKDIMZ];
    __shared__ T m0[BLOCKDIMZ];

    __shared__ T ld[BLOCKDIMY];
    __shared__ T md[BLOCKDIMY];


    int i;

    // TODO. Using 3 times more shared memory than we
    // really require here, since there's only
    // one wavelength per channel.
    if(threadIdx.y == 0 && threadIdx.z == 0)
    {
        wl[threadIdx.x] = wavelength[POLCHAN >> 2];
    }

    // LM coordinates vary by source only,
    // not antenna or polarised channel
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        i = SRC;   l0[threadIdx.z] = lm[i];
        i += NSRC; m0[threadIdx.z] = lm[i];
    }

    __syncthreads();

    for(int TIME=0; TIME < NTIME; ++TIME)
    {
        // Pointing errors vary by time and antenna
        if(threadIdx.z == 0 && threadIdx.x == 0)
        {
            i = TIME*NA + ANT; ld[threadIdx.y] = point_errors[i];
            i += NTIME*NA;     md[threadIdx.y] = point_errors[i];
        }

        __syncthreads();

        // Figure out how far the source has
        // rotated within the beam
        T sint, cost;
        Po::sincos(beam_rot_vel*TIME, &sint, &cost);

        // Rotate the source
        T l = l0[threadIdx.z]*cost - m0[threadIdx.z]*sint;
        T m = l0[threadIdx.z]*sint + m0[threadIdx.z]*cost;

        // Add the pointing errors for this antenna.
        l += ld[threadIdx.y];
        m += lm[threadIdx.y];

    }
}

extern "C" {

#define stamp_jones_E_beam_fn(ft,ct) \
__global__ void \
rime_jones_E_beam_ ## ft( \
    ft * lm, \
    ft * wavelength, \
    ft * point_errors, \
    ct * E_beam, \
    ft beam_rot_vel) \
{ \
    rime_jones_E_beam_impl<ft>( \
        lm, wavelength, point_errors, E_beam,
        beam_rot_vel); \
}

stamp_jones_E_beam_fn(float,float2);
stamp_jones_E_beam_fn(double,double2);

} // extern "C" {
""")

class RimeEBeam(Node):
    def __init__(self):
        super(RimeEBeam, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        self.polchans = 4*slvr.nchan

        # Get a property dictionary off the solver
        D = slvr.get_properties()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        # Update kernel parameters to cater for radically
        # smaller problem sizes. Caters for a subtle bug
        # with Kepler shuffles and warp sizes < 32
        if self.polchans < D['BLOCKDIMX']: D['BLOCKDIMX'] = self.polchans
        if slvr.na < D['BLOCKDIMY']: D['BLOCKDIMY'] = slvr.na
        if slvr.nsrc < D['BLOCKDIMZ']: D['BLOCKDIMZ'] = slvr.nsrc

        regs = str(FLOAT_PARAMS['maxregs'] \
                if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_jones_E_beam_float' \
            if slvr.is_float() is True else \
            'rime_jones_E_beam_double'

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
        polchans_per_block = D['BLOCKDIMX']
        ants_per_block = D['BLOCKDIMY']
        srcs_per_block = D['BLOCKDIMZ']

        polchan_blocks = self.blocks_required(self.polchans, polchans_per_block)
        ant_blocks = self.blocks_required(slvr.na, ants_per_block)
        src_blocks = self.blocks_required(slvr.nsrc, srcs_per_block)

        return {
            'block' : (polchans_per_block, ants_per_block, srcs_per_block),
            'grid'  : (polchan_blocks, ant_blocks, src_blocks),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        self.kernel(slvr.lm_gpu, slvr.wavelength_gpu,
            self.point_errors_gpu, slvr.E_beam_gpu,
            slvr.beam_rot_vel,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
