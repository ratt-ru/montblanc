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
    'BLOCKDIMY' : 32,   # Number of antenna
    'BLOCKDIMZ' : 1,    #
    'maxregs'   : 48    # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels and polarisations
    'BLOCKDIMY' : 16,   # Number of antenna
    'BLOCKDIMZ' : 1,    #
    'maxregs'   : 63    # Maximum number of registers
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
#define LEXT(name) (C.name.lower_extent)
#define UEXT(name) (C.name.upper_extent)
#define DEXT(name) (C.name.upper_extent - C.name.lower_extent)
#define GLOBAL(name) (C.name.global_size)
#define LOCAL(name) (C.name.local_size)

#define NA LOCAL(na)
#define NBL LOCAL(nbl)
#define NCHAN LOCAL(nchan)
#define NTIME LOCAL(ntime)
#define NSRC LOCAL(nsrc)
#define NPOL LOCAL(npol)
#define NPOLCHAN LOCAL(npolchan)
#define BEAM_LW LOCAL(beam_lw)
#define BEAM_MH LOCAL(beam_mh)
#define BEAM_NUD LOCAL(beam_nud)

template <typename T> __device__ __forceinline__
int ebeam_pol()
    { return threadIdx.x & 0x3; }

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__
void trilinear_interpolate(
    typename Tr::ct & sum,
    typename Tr::ft & abs_sum,
    typename Tr::ct * E_beam,
    float gl, float gm, float gchan,
    const T & weight)
{
    // If this source is outside the cube, do nothing
    if(gl < 0 || gl >= BEAM_LW || gm < 0 || gm >= BEAM_MH)
        { return; }

    int i = ((int(gl)*BEAM_MH +
        int(gm))*BEAM_NUD +
        int(gchan))*NPOL + ebeam_pol<T>();

    // Perhaps unnecessary as long as BLOCKDIMX is 32
    typename Tr::ct pol = cub::ThreadLoad<cub::LOAD_LDG>(E_beam + i);
    sum.x += weight*pol.x;
    sum.y += weight*pol.y;
    abs_sum += weight*Po::abs(pol);
}

template <typename T> class EBeamTraits {};

template <> class EBeamTraits<float>
{
public:
    typedef float2 LMType;
    typedef float2 PointErrorType;
    typedef float2 AntennaScaleType;
};

template <> class EBeamTraits<double>
{
public:
    typedef double2 LMType;
    typedef double2 PointErrorType;
    typedef double2 AntennaScaleType;
};

template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones_E_beam_impl(
    typename EBeamTraits<T>::LMType * lm,
    typename EBeamTraits<T>::PointErrorType * point_errors,
    typename EBeamTraits<T>::AntennaScaleType * antenna_scaling,
    typename Tr::ct * E_beam,
    typename Tr::ct * jones,
    T parallactic_angle,
    T beam_ll, T beam_lm,
    T beam_ul, T beam_um)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;
    #define POL (threadIdx.x & 0x3)
    #define BLOCKCHANS (BLOCKDIMX >> 2)

    if(SRC >= DEXT(nsrc) || ANT >= DEXT(na) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ typename EBeamTraits<T>::LMType s_lm0[BLOCKDIMZ];
    __shared__ typename EBeamTraits<T>::PointErrorType s_lmd[BLOCKDIMY][BLOCKCHANS];
    __shared__ typename EBeamTraits<T>::AntennaScaleType s_ab[BLOCKDIMY][BLOCKCHANS];

    int i;

    // LM coordinates vary by source only,
    // not antenna or polarised channel
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        i = SRC;   s_lm0[threadIdx.z] = lm[i];
    }

    // Antenna scaling factors vary by antenna and channel,
    // but not source or timestep
    if(threadIdx.z == 0 && (threadIdx.x & 0x3) == 0)
    {
        int blockchan = threadIdx.x >> 2;
        i = ANT*NCHAN + (POLCHAN >> 2);
        s_ab[threadIdx.y][blockchan] = antenna_scaling[i];
    }

    __syncthreads();

    for(int TIME=0; TIME < DEXT(ntime); ++TIME)
    {
        // Pointing errors vary by time, antenna and channel,
        // but not source
        if(threadIdx.z == 0 && (threadIdx.x & 0x3) == 0)
        {
            int blockchan = threadIdx.x >> 2;
            i = (TIME*NA + ANT)*NCHAN + (POLCHAN >> 2);
            s_lmd[threadIdx.y][blockchan] = point_errors[i];
        }

        __syncthreads();

        // Figure out how far the source has
        // rotated within the beam
        T sint, cost;
        Po::sincos(parallactic_angle*TIME, &sint, &cost);

        // Rotate the source
        T l = s_lm0[threadIdx.z].x*cost - s_lm0[threadIdx.z].y*sint;
        T m = s_lm0[threadIdx.z].x*sint + s_lm0[threadIdx.z].y*cost;

        // Add the pointing errors for this antenna.
        int blockchan = threadIdx.x >> 2;
        l += s_lmd[threadIdx.y][blockchan].x;
        m += s_lmd[threadIdx.y][blockchan].y;

        // Multiply by the antenna scaling factors.
        l *= s_ab[threadIdx.y][blockchan].x;
        m *= s_ab[threadIdx.y][blockchan].y;

        // Compute grid position and difference from
        // actual position for the source at each channel
        l = T(BEAM_LW-1) * (l - beam_ll) / (beam_ul - beam_ll);
        float gl = floorf(l);
        float ld = l - gl;

        m = T(BEAM_MH-1) * (m - beam_lm) / (beam_um - beam_lm);
        float gm = floorf(m);
        float md = m - gm;

        // Work out where we are in the beam cube.
        // POLCHAN >> 2 is our position in the local channel space
        // Add this to the lower extent in the global channel space
        float chan = float(BEAM_NUD-1) * float(POLCHAN>>2 + LEXT(nchan))
            / float(GLOBAL(nchan));
        float gchan = floorf(chan);
        float chd = chan - gchan;

        typename Tr::ct sum = Po::make_ct(0.0, 0.0);
        typename Tr::ft abs_sum = T(0.0);

        // A simplified bilinear weighting is used here. Given
        // point x between points x1 and x2, with function f
        // provided values f(x1) and f(x2) at these points.
        //
        // x1 ------- x ---------- x2
        //
        // Then, the value of f can be approximated using the following:
        // f(x) ~= f(x1)(x2-x)/(x2-x1) + f(x2)(x-x1)/(x2-x1)
        //
        // Note how the value f(x1) is weighted with the distance
        // from the opposite point (x2-x).
        //
        // As we are interpolating on a grid, we have the following
        // 1. (x2 - x1) == 1
        // 2. (x - x1)  == 1 - 1 + (x - x1)
        //              == 1 - (x2 - x1) + (x - x1)
        //              == 1 - (x2 - x)
        // 2. (x2 - x)  == 1 - 1 + (x2 - x)
        //              == 1 - (x2 - x1) + (x2 - x)
        //              == 1 - (x - x1)
        //
        // Extending the above to 3D, we have
        // f(x,y,z) ~= f(x1,y1,z1)(x2-x)(y2-y)(z2-z) + ...
        //           + f(x2,y2,z2)(x-x1)(y-y1)(z-z1)
        //
        // f(x,y,z) ~= f(x1,y1,z1)(1-(x-x1))(1-(y-y1))(1-(z-z1)) + ...
        //           + f(x2,y2,z2)   (x-x1)    (y-y1)    (z-z1)

        // Load in the complex values from the E beam
        // at the supplied coordinate offsets.
        // Save the sum of abs in sum.real
        // and the sum of args in sum.imag
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 0.0f, gm + 0.0f, gchan + 0.0f,
            (1.0f-ld)*(1.0f-md)*(1.0f-chd));
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 1.0f, gm + 0.0f, gchan + 0.0f,
            ld*(1.0f-md)*(1.0f-chd));
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 0.0f, gm + 1.0f, gchan + 0.0f,
            (1.0f-ld)*md*(1.0f-chd));
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 1.0f, gm + 1.0f, gchan + 0.0f,
            ld*md*(1.0f-chd));

        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 0.0f, gm + 0.0f, gchan + 1.0f,
            (1.0f-ld)*(1.0f-md)*chd);
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 1.0f, gm + 0.0f, gchan + 1.0f,
            ld*(1.0f-md)*chd);
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 0.0f, gm + 1.0f, gchan + 1.0f,
            (1.0f-ld)*md*chd);
        trilinear_interpolate<T>(sum, abs_sum, E_beam,
            gl + 1.0f, gm + 1.0f, gchan + 1.0f,
            ld*md*chd);

        // Determine the normalised angle
        typename Tr::ft angle = Po::arg(sum);

        // Take the complex exponent of the angle
        // and multiply by the sum of abs
        typename Tr::ct value;
        Po::sincos(angle, &value.y, &value.x);
        value.x *= abs_sum;
        value.y *= abs_sum;

        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        jones[i] = value;
        __syncthreads();
    }
}

extern "C" {

#define stamp_jones_E_beam_fn(ft,ct,lm_type,pe_type,as_type) \
__global__ void \
rime_jones_E_beam_ ## ft( \
    lm_type * lm, \
    pe_type * point_errors, \
    as_type * antenna_scaling, \
    ct * E_beam, \
    ct * jones, \
    ft parallactic_angle, \
    ft beam_ll, ft beam_lm, \
    ft beam_ul, ft beam_um) \
{ \
    rime_jones_E_beam_impl<ft>( \
        lm, point_errors, antenna_scaling, E_beam, jones, \
        parallactic_angle, beam_ll, beam_lm, beam_ul, beam_um); \
}

stamp_jones_E_beam_fn(float,float2,float2,float2,float2);
stamp_jones_E_beam_fn(double,double2,double2,double2,double2);

} // extern "C" {
""")

class RimeEBeam(Node):
    def __init__(self):
        super(RimeEBeam, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver
        nsrc, na, npolchan = slvr.dim_local_size('nsrc', 'na', 'npolchan')

        # Get a property dictionary off the solver
        D = slvr.template_dict()
        # Include our kernel parameters
        D.update(FLOAT_PARAMS if slvr.is_float() else DOUBLE_PARAMS)
        D['rime_const_data_struct'] = slvr.const_data().string_def()

        D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'] = \
            mbu.redistribute_threads(
                D['BLOCKDIMX'], D['BLOCKDIMY'], D['BLOCKDIMZ'],
                npolchan, na, nsrc)

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

        self.rime_const_data = self.mod.get_global('C')
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

        nsrc, na, npolchan = slvr.dim_local_size('nsrc', 'na', 'npolchan')
        polchan_blocks = mbu.blocks_required(npolchan, polchans_per_block)
        ant_blocks = mbu.blocks_required(na, ants_per_block)
        src_blocks = mbu.blocks_required(nsrc, srcs_per_block)

        return {
            'block' : (polchans_per_block, ants_per_block, srcs_per_block),
            'grid'  : (polchan_blocks, ant_blocks, src_blocks),
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

        self.kernel(slvr.lm,
            slvr.point_errors, slvr.antenna_scaling,
            slvr.E_beam, slvr.jones,
            slvr.parallactic_angle,
            slvr.beam_ll, slvr.beam_lm,
            slvr.beam_ul, slvr.beam_um,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
