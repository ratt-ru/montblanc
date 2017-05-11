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

# We put some blocks in the time dimension because
# radio sources should exhibit spatial locality
# as they are rotated by parallactic angle
# There should therefore be an opportunity to
# take advantage of Kepler's L1 Texture Cache
# (See the cub::LOAD_LDG)

FLOAT_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels x polarisations
    'BLOCKDIMY' : 8,    # Number of antenna
    'BLOCKDIMZ' : 4,    # Number of timesteps
    'maxregs'   : 48    # Maximum number of registers
}

DOUBLE_PARAMS = {
    'BLOCKDIMX' : 32,   # Number of channels x polarisations
    'BLOCKDIMY' : 8,    # Number of antenna
    'BLOCKDIMZ' : 4,    # Number of timesteps
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

// Infer channel from x thread
 __device__ __forceinline__ int thread_chan()
    { return threadIdx.x >> 2; }

// Infer polarisation from x thread
__device__ __forceinline__ int ebeam_pol()
    { return threadIdx.x & 0x3; }


template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> > 
__device__ __forceinline__
void trilinear_interpolate(
    typename Tr::ct & pol_sum,
    typename Tr::ft & abs_sum,
    typename Tr::ct * E_beam,
    const T & gl,
    const T & gm,
    const T & gchan,
    const T & weight)
{
    int i = ((int(gl)*BEAM_MH + int(gm))*BEAM_NUD + int(gchan))*NPOL +ebeam_pol();

    // Perhaps unnecessary as long as BLOCKDIMX is 32
    typename Tr::ct pol = cub::ThreadLoad<cub::LOAD_LDG>(E_beam + i);
    pol_sum.x += weight*pol.x;
    pol_sum.y += weight*pol.y;
    abs_sum += weight*Po::abs(pol);
}


template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> > 
__device__ __forceinline__
void trilinear_interpolate_J1(
    typename Tr::ct & pol_sum,
    typename Tr::ft & abs_sum,
    typename Tr::ct * E_beam,
    const T & gl,
    const T & gm,
    const T & gchan,
    const T & weight)
{
    int i = ((int(gl)*BEAM_MH + int(gm))*BEAM_NUD + int(gchan));

    // Perhaps unnecessary as long as BLOCKDIMX is 32
    typename Tr::ct pol = cub::ThreadLoad<cub::LOAD_LDG>(E_beam + i);
    pol_sum.x += weight*pol.x;
    pol_sum.y += weight*pol.y;
    abs_sum += weight*Po::abs(pol);
}


template <typename T> class EBeamTraits {};

template <> class EBeamTraits<float>
{
public:
    typedef float2 LMType;
    typedef float ParallacticAngleType;
    typedef float2 PointErrorType;
    typedef float2 AntennaScaleType;
};

template <> class EBeamTraits<double>
{
public:
    typedef double2 LMType;
    typedef double ParallacticAngleType;
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
    typename EBeamTraits<T>::ParallacticAngleType * parallactic_angles,
    typename EBeamTraits<T>::PointErrorType * point_errors,
    typename EBeamTraits<T>::AntennaScaleType * antenna_scaling,
    typename Tr::ft * frequency,
    typename Tr::ct * E_beam,
    typename Tr::ct * jones,
    T beam_ll, T beam_lm, T beam_lfreq,
    T beam_ul, T beam_um, T beam_ufreq)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;
    #define BLOCKCHANS (BLOCKDIMX >> 2)

    typedef typename EBeamTraits<T>::LMType LMType;
    typedef typename EBeamTraits<T>::PointErrorType PointErrorType;
    typedef typename EBeamTraits<T>::AntennaScaleType AntennaScaleType;

    if(TIME >= DEXT(ntime) || ANT >= DEXT(na) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ struct {
        T lscale;             // l axis scaling factor
        T mscale;             // m axis scaling factor
        T pa_sin[BLOCKDIMZ][BLOCKDIMY];  // sin of parallactic angle
        T pa_cos[BLOCKDIMZ][BLOCKDIMY];  // cos of parallactic angle
        T gchan0[BLOCKCHANS];  // channel grid position (snapped)
        T gchan1[BLOCKCHANS];  // channel grid position (snapped)
        T chd[BLOCKCHANS];    // difference between gchan0 and actual grid position
        PointErrorType pe[BLOCKDIMZ][BLOCKDIMY][BLOCKCHANS];  // pointing errors
        AntennaScaleType as[BLOCKDIMY][BLOCKCHANS];           // antenna scaling
    } shared;


    int i;

    // Precompute l and m scaling factors in shared memory
    if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        shared.lscale = T(BEAM_LW - 1) / (beam_ul - beam_ll);
        shared.mscale = T(BEAM_MH - 1) / (beam_um - beam_lm);
    }

    // Pointing errors vary by time, antenna and channel,
    if(ebeam_pol() == 0)
    {
        i = (TIME*NA + ANT)*NCHAN + (POLCHAN >> 2);
        shared.pe[threadIdx.z][threadIdx.y][thread_chan()] = point_errors[i];
    }

    // Antenna scaling factors vary by antenna and channel,
    // but not timestep
    if(threadIdx.z == 0 && ebeam_pol() == 0)
    {
        i = ANT*NCHAN + (POLCHAN >> 2);
        shared.as[threadIdx.y][thread_chan()] = antenna_scaling[i];
    }

    // Frequency vary by channel, but not timestep or antenna
    if(threadIdx.z == 0 && threadIdx.y == 0 && ebeam_pol() == 0)
    {
        // Channel coordinate
        // channel grid position
        T freqscale = T(BEAM_NUD-1) / (beam_ufreq - beam_lfreq);
        T chan = freqscale * (frequency[POLCHAN >> 2] - beam_lfreq);
        // clamp to grid edges
        chan = Po::clamp(chan, 0.0, BEAM_NUD-1);
        // Snap to grid coordinate
        shared.gchan0[thread_chan()] = Po::floor(chan);
        shared.gchan1[thread_chan()] = Po::min(
            shared.gchan0[thread_chan()] + 1.0, BEAM_NUD-1);
        // Offset of snapped coordinate from grid position
        shared.chd[thread_chan()] = chan - shared.gchan0[thread_chan()];
    }

    // Parallactic angles vary by time and antenna, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*NA + ANT;
        T parangle = parallactic_angles[i];
        Po::sincos(parangle,
            &shared.pa_sin[threadIdx.z][threadIdx.y],
            &shared.pa_cos[threadIdx.z][threadIdx.y]);
    }

    __syncthreads();

    // Loop over sources
    for(int SRC=0; SRC < DEXT(nsrc); ++SRC)
    {
        // lm coordinate for this source
        LMType rlm = lm[SRC];

        // L coordinate
        // Rotate
        T l = rlm.x*shared.pa_cos[threadIdx.z][threadIdx.y] -
            rlm.y*shared.pa_sin[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        l += shared.pe[threadIdx.z][threadIdx.y][thread_chan()].x;
        // Scale by antenna scaling factors
        l *= shared.as[threadIdx.y][thread_chan()].x;
        // l grid position
        l = shared.lscale * (l - beam_ll);
        // clamp to grid edges
        l = Po::clamp(0.0, l, BEAM_LW-1);
        // Snap to grid coordinate
        T gl0 = Po::floor(l);
        T gl1 = Po::min(gl0 + 1.0, BEAM_LW-1);
        // Offset of snapped coordinate from grid position
        T ld = l - gl0;

        // M coordinate
        // rotate
        T m = rlm.x*shared.pa_sin[threadIdx.z][threadIdx.y] +
            rlm.y*shared.pa_cos[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        m += shared.pe[threadIdx.z][threadIdx.y][thread_chan()].y;
        // Scale by antenna scaling factors
        m *= shared.as[threadIdx.y][thread_chan()].y;
        // m grid position
        m = shared.mscale * (m - beam_lm);
        // clamp to grid edges
        m = Po::clamp(0.0, m, BEAM_MH-1);
        // Snap to grid position
        T gm0 = Po::floor(m);
        T gm1 = Po::min(gm0 + 1.0, BEAM_MH-1);
        // Offset of snapped coordinate from grid position
        T md = m - gm0;

        typename Tr::ct pol_sum = Po::make_ct(0.0, 0.0);
        typename Tr::ft abs_sum = T(0.0);

        // A simplified trilinear weighting is used here. Given
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
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl0, gm0, shared.gchan0[thread_chan()],
            (1.0f-ld)*(1.0f-md)*(1.0f-shared.chd[thread_chan()]));
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl1, gm0, shared.gchan0[thread_chan()],
            ld*(1.0f-md)*(1.0f-shared.chd[thread_chan()]));
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl0, gm1, shared.gchan0[thread_chan()],
            (1.0f-ld)*md*(1.0f-shared.chd[thread_chan()]));
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl1, gm1, shared.gchan0[thread_chan()],
            ld*md*(1.0f-shared.chd[thread_chan()]));

        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl0, gm0, shared.gchan1[thread_chan()],
            (1.0f-ld)*(1.0f-md)*shared.chd[thread_chan()]);
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl1, gm0, shared.gchan1[thread_chan()],
            ld*(1.0f-md)*shared.chd[thread_chan()]);
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl0, gm1, shared.gchan1[thread_chan()],
            (1.0f-ld)*md*shared.chd[thread_chan()]);
        trilinear_interpolate<T>(pol_sum, abs_sum, E_beam,
            gl1, gm1, shared.gchan1[thread_chan()],
            ld*md*shared.chd[thread_chan()]);

        // Normalise the angle and multiply in the absolute sum
        typename Tr::ft norm = Po::rsqrt(pol_sum.x*pol_sum.x + pol_sum.y*pol_sum.y);
        if(!::isfinite(norm))
            { norm = 1.0; }

        pol_sum.x *= norm * abs_sum;
        pol_sum.y *= norm * abs_sum;

        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        jones[i] = pol_sum;
    }
}



template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__
void rime_jones1_E_beam_impl(
    typename EBeamTraits<T>::LMType * lm,
    typename EBeamTraits<T>::ParallacticAngleType * parallactic_angles,
    typename EBeamTraits<T>::PointErrorType * point_errors,
    typename EBeamTraits<T>::AntennaScaleType * antenna_scaling,
    typename Tr::ft * frequency,
    typename Tr::ct * E_beam,
    typename Tr::ct * jones,
    T beam_ll, T beam_lm, T beam_lfreq,
    T beam_ul, T beam_um, T beam_ufreq)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    typedef typename EBeamTraits<T>::LMType LMType;
    typedef typename EBeamTraits<T>::PointErrorType PointErrorType;
    typedef typename EBeamTraits<T>::AntennaScaleType AntennaScaleType;

    if(TIME >= DEXT(ntime) || ANT >= DEXT(na) || POLCHAN >= DEXT(npolchan))
        return;

    __shared__ struct {
        T lscale;             // l axis scaling factor
        T mscale;             // m axis scaling factor
        T pa_sin[BLOCKDIMZ][BLOCKDIMY];  // sin of parallactic angle
        T pa_cos[BLOCKDIMZ][BLOCKDIMY];  // cos of parallactic angle
        T gchan0[BLOCKDIMX];  // channel grid position (snapped)
        T gchan1[BLOCKDIMX];  // channel grid position (snapped)
        T chd[BLOCKDIMX];    // difference between gchan0 and actual grid position
        PointErrorType pe[BLOCKDIMZ][BLOCKDIMY][BLOCKDIMX];  // pointing errors
        AntennaScaleType as[BLOCKDIMY][BLOCKDIMX];           // antenna scaling
    } shared;


    int i;

    // Precompute l and m scaling factors in shared memory
    if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        shared.lscale = T(BEAM_LW - 1) / (beam_ul - beam_ll);
        shared.mscale = T(BEAM_MH - 1) / (beam_um - beam_lm);
    }

    // Pointing errors vary by time, antenna and channel,
    i = (TIME*NA + ANT)*NCHAN + POLCHAN;
    shared.pe[threadIdx.z][threadIdx.y][threadIdx.x] = point_errors[i];

    // Antenna scaling factors vary by antenna and channel,
    // but not timestep
    if(threadIdx.z == 0)
    {
        i = ANT*NCHAN + POLCHAN;
        shared.as[threadIdx.y][threadIdx.x] = antenna_scaling[i];
    }

    // Frequency vary by channel, but not timestep or antenna
    if(threadIdx.z == 0 && threadIdx.y == 0)
    {
        // Channel coordinate
        // channel grid position
        T freqscale = T(BEAM_NUD-1) / (beam_ufreq - beam_lfreq);
        T chan = freqscale * (frequency[POLCHAN] - beam_lfreq);
        // clamp to grid edges
        chan = Po::clamp(chan, 0.0, BEAM_NUD-1);
        // Snap to grid coordinate
        shared.gchan0[threadIdx.x] = Po::floor(chan);
        shared.gchan1[threadIdx.x] = Po::min(
            shared.gchan0[threadIdx.x] + 1.0, BEAM_NUD-1);
        // Offset of snapped coordinate from grid position
        shared.chd[threadIdx.x] = chan - shared.gchan0[threadIdx.x];
    }

    // Parallactic angles vary by time and antenna, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*NA + ANT;
        T parangle = parallactic_angles[i];
        Po::sincos(parangle,
            &shared.pa_sin[threadIdx.z][threadIdx.y],
            &shared.pa_cos[threadIdx.z][threadIdx.y]);
    }

    __syncthreads();

    // Loop over sources
    for(int SRC=0; SRC < DEXT(nsrc); ++SRC)
    {
        // lm coordinate for this source
        LMType rlm = lm[SRC];

        // L coordinate
        // Rotate
        T l = rlm.x*shared.pa_cos[threadIdx.z][threadIdx.y] -
            rlm.y*shared.pa_sin[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        l += shared.pe[threadIdx.z][threadIdx.y][threadIdx.x].x;
        // Scale by antenna scaling factors
        l *= shared.as[threadIdx.y][threadIdx.x].x;
        // l grid position
        l = shared.lscale * (l - beam_ll);
        // clamp to grid edges
        l = Po::clamp(0.0, l, BEAM_LW-1);
        // Snap to grid coordinate
        T gl0 = Po::floor(l);
        T gl1 = Po::min(gl0 + 1.0, BEAM_LW-1);
        // Offset of snapped coordinate from grid position
        T ld = l - gl0;

        // M coordinate
        // rotate
        T m = rlm.x*shared.pa_sin[threadIdx.z][threadIdx.y] +
            rlm.y*shared.pa_cos[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        m += shared.pe[threadIdx.z][threadIdx.y][threadIdx.x].y;
        // Scale by antenna scaling factors
        m *= shared.as[threadIdx.y][threadIdx.x].y;
        // m grid position
        m = shared.mscale * (m - beam_lm);
        // clamp to grid edges
        m = Po::clamp(0.0, m, BEAM_MH-1);
        // Snap to grid position
        T gm0 = Po::floor(m);
        T gm1 = Po::min(gm0 + 1.0, BEAM_MH-1);
        // Offset of snapped coordinate from grid position
        T md = m - gm0;

        typename Tr::ct pol_sum = Po::make_ct(0.0, 0.0);
        typename Tr::ft abs_sum = T(0.0);

        // A simplified trilinear weighting is used here. Given
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
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl0, gm0, shared.gchan0[threadIdx.x],
            (1.0f-ld)*(1.0f-md)*(1.0f-shared.chd[threadIdx.x]));
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl1, gm0, shared.gchan0[threadIdx.x],
            ld*(1.0f-md)*(1.0f-shared.chd[threadIdx.x]));
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl0, gm1, shared.gchan0[threadIdx.x],
            (1.0f-ld)*md*(1.0f-shared.chd[threadIdx.x]));
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl1, gm1, shared.gchan0[threadIdx.x],
            ld*md*(1.0f-shared.chd[threadIdx.x]));

        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl0, gm0, shared.gchan1[threadIdx.x],
            (1.0f-ld)*(1.0f-md)*shared.chd[threadIdx.x]);
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl1, gm0, shared.gchan1[threadIdx.x],
            ld*(1.0f-md)*shared.chd[threadIdx.x]);
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl0, gm1, shared.gchan1[threadIdx.x],
            (1.0f-ld)*md*shared.chd[threadIdx.x]);
        trilinear_interpolate_J1<T>(pol_sum, abs_sum, E_beam,
            gl1, gm1, shared.gchan1[threadIdx.x],
            ld*md*shared.chd[threadIdx.x]);

        // Normalise the angle and multiply in the absolute sum
        typename Tr::ft norm = Po::rsqrt(pol_sum.x*pol_sum.x + pol_sum.y*pol_sum.y);
        if(!::isfinite(norm))
            { norm = 1.0; }

        pol_sum.x *= norm * abs_sum;
        pol_sum.y *= norm * abs_sum;

        i = ((SRC*NTIME + TIME)*NA + ANT)*NPOLCHAN + POLCHAN;
        jones[i] = pol_sum;
    }
}





extern "C" {

#define stamp_jones_E_beam_fn(ft,ct,lm_type,pa_type,pe_type,as_type) \
__global__ void \
rime_jones_E_beam_ ## ft( \
    lm_type * lm, \
    pa_type * parallactic_angles, \
    pe_type * point_errors, \
    as_type * antenna_scaling, \
    ft * frequency, \
    ct * E_beam, \
    ct * jones, \
    ft beam_ll, ft beam_lm, ft beam_lfreq, \
    ft beam_ul, ft beam_um, ft beam_ufreq) \
{ \
    rime_jones_E_beam_impl<ft>( \
        lm, parallactic_angles, point_errors, \
        antenna_scaling, frequency, E_beam, jones, \
        beam_ll, beam_lm, beam_lfreq, \
        beam_ul, beam_um, beam_ufreq); \
}

#define stamp_jones1_E_beam_fn(ft,ct,lm_type,pa_type,pe_type,as_type) \
__global__ void \
rime_jones1_E_beam_ ## ft( \
    lm_type * lm, \
    pa_type * parallactic_angles, \
    pe_type * point_errors, \
    as_type * antenna_scaling, \
    ft * frequency, \
    ct * E_beam, \
    ct * jones, \
    ft beam_ll, ft beam_lm, ft beam_lfreq, \
    ft beam_ul, ft beam_um, ft beam_ufreq) \
{ \
    rime_jones1_E_beam_impl<ft>( \
        lm, parallactic_angles, point_errors, \
        antenna_scaling, frequency, E_beam, jones, \
        beam_ll, beam_lm, beam_lfreq, \
        beam_ul, beam_um, beam_ufreq); \
}

${stamp_function}

} // extern "C" {
""")

class RimeEBeam(Node):
    def __init__(self):
        super(RimeEBeam, self).__init__()

    def initialise(self, solver, stream=None):
        slvr = solver
        ntime, na, npolchan, npol = slvr.dim_local_size('ntime', 'na', 'npolchan', 'npol')

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

        # Create the signature of the call to the function stamping macro
        stamp_args = ', '.join([
            'float' if slvr.is_float() else 'double',
            'float2' if slvr.is_float() else 'double2',
            'float2' if slvr.is_float() else 'double2',
            'float' if slvr.is_float() else 'double',
            'float2' if slvr.is_float() else 'double2',
            'float2' if slvr.is_float() else 'double2'])

        if npol == 4:
            stamp_fn = ''.join(['stamp_jones_E_beam_fn(', stamp_args, ')'])
            D['stamp_function'] = stamp_fn
            kname = 'rime_jones_E_beam_float' \
                if slvr.is_float() is True else \
                'rime_jones_E_beam_double'

        if npol == 1:
            stamp_fn = ''.join(['stamp_jones1_E_beam_fn(', stamp_args, ')'])
            D['stamp_function'] = stamp_fn
            kname = 'rime_jones1_E_beam_float' \
                if slvr.is_float() is True else \
                'rime_jones1_E_beam_double'

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
                self.rime_const_data[0],
                slvr.const_data().ndary(),
                stream=stream)
        else:
            cuda.memcpy_htod(
                self.rime_const_data[0],
                slvr.const_data().ndary())

        self.kernel(slvr.lm, slvr.parallactic_angles,
            slvr.point_errors, slvr.antenna_scaling, slvr.frequency,
            slvr.E_beam, slvr.jones,
            slvr.beam_ll, slvr.beam_lm, slvr.beam_lfreq,
            slvr.beam_ul, slvr.beam_um, slvr.beam_ufreq,
            stream=stream, **self.launch_params)

    def post_execution(self, solver, stream=None):
        pass
