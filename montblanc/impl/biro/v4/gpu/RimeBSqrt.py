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
    'BLOCKDIMX' : 32,   # Number of polarisations and timesteps
    'BLOCKDIMY' : 8,    # Number of sources
    'BLOCKDIMZ' : 1,    #
    'maxregs'   : 32    # Maximum number of registers
}

# 44 registers results in some spillage into
# local memory
DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of polarisations and timesteps
    'BLOCKDIMY' : 4,    # Number of sources
    'BLOCKDIMZ' : 1,    #
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
#define NPOLTIME (NPOL*NTIME)

#define BLOCKDIMX (${BLOCKDIMX})
#define BLOCKDIMY (${BLOCKDIMY})
#define BLOCKDIMZ (${BLOCKDIMZ})

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_B_sqrt_impl(
    T * stokes,
    T * alpha,
    T * wavelength,
    typename Tr::ct * B_sqrt)
{
    int POLTIME = blockIdx.x*blockDim.x + threadIdx.x;
    int SRC = blockIdx.y*blockDim.y + threadIdx.y;
    #define POL (threadIdx.x & 0x3)

    if(SRC >= NSRC || POLTIME >= NPOLTIME)
        return;

    int i = SRC*NPOLTIME + POLTIME;
    typename Tr::ft pol = stokes[i];
    typename Tr::ct brightness;

    montblanc::create_brightness<T>(brightness, pol);

    // Write out the phase terms
    B_sqrt[i] = brightness;
}

extern "C" {

#define stamp_jones_B_sqrt_fn(ft,ct) \
__global__ void \
rime_jones_B_sqrt_ ## ft( \
    ft * stokes, \
    ft * alpha, \
    ft * wavelength, \
    ct * B_sqrt) \
{ \
    rime_jones_B_sqrt_impl<ft>(stokes, alpha, \
        wavelength, B_sqrt); \
}

stamp_jones_B_sqrt_fn(float,float2);
stamp_jones_B_sqrt_fn(double,double2);

} // extern "C" {
""")

class RimeBSqrt(Node):
    def __init__(self):
        super(RimeBSqrt, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver

        self.pol_times = 4*slvr.ntime

        # Get a property dictionary off the solver
        D = slvr.get_properties()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)

        # Update kernel parameters to cater for radically
        # smaller problem sizes. Caters for a subtle bug
        # with Kepler shuffles and warp sizes < 32
        if self.pol_times < D['BLOCKDIMX']:
            D['BLOCKDIMX'] = self.pol_times
        if slvr.nsrc < D['BLOCKDIMY']:
            D['BLOCKDIMY'] = slvr.nsrc

        regs = str(FLOAT_PARAMS['maxregs'] \
                if slvr.is_float() else DOUBLE_PARAMS['maxregs'])

        kname = 'rime_jones_B_sqrt_float' \
            if slvr.is_float() is True else \
            'rime_jones_B_sqrt_double'

        kernel_string = KERNEL_TEMPLATE.substitute(**D)

        with open('kernel_debug.cu', 'w') as f:
            f.write(kernel_string)
            f.close()

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
        pol_times_per_block = D['BLOCKDIMX']
        srcs_per_block = D['BLOCKDIMY']

        pol_time_blocks = self.blocks_required(self.pol_times, pol_times_per_block)
        src_blocks = self.blocks_required(slvr.nsrc, srcs_per_block)

        return {
            'block' : (pol_times_per_block, srcs_per_block, 1),
            'grid'  : (pol_time_blocks, src_blocks, 1),
        }

    def execute(self, solver, stream=None):
        slvr = solver

        self.kernel(slvr.stokes_gpu, slvr.alpha_gpu,
            slvr.wavelength_gpu, slvr.B_sqrt_gpu,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
