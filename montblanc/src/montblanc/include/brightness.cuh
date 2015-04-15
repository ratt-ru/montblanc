// Copyright (c) 2015 Simon Perkins
//
// This file is part of montblanc.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.

#ifndef _MONTBLANC_BRIGHTNESS_CUH
#define _MONTBLANC_BRIGHTNESS_CUH

#include <cub/cub/cub.cuh>
#include <montblanc/include/abstraction.cuh>

namespace montblanc {

// Create the mask for transforming the second polarisation
// Given:
//   thread 0 : pol = I;
//   thread 1 : pol = Q;
//   thread 2 : pol = U;
//   thread 3 : pol = V;
// we want to obtain the following brightness matrix:
//   thread 0 : B = I+Q;
//   thread 1 : B = U+iV;
//   thread 2 : B = U-iV;
//   thread 1 : B = I-Q;
// The second values (+Q, +iV,-iV, -Q) in the above expression must
// be transformed with the following complex mask.
//   thread 0 :  1 * (1 + 0j)
//   thread 1 :  1 * (0 + 1j)
//   thread 2 : -1 * (0 + 1j)
//   thread 3 : -1 * (1 + 0j)
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__ void create_brightness_mask(typename Tr::ct & mask)
{
    int sign = ((int(cub::LaneId()) - 2) & 0x2) - 1;
    mask.x = T(sign*((int(cub::LaneId()) - 1) & 0x2) >> 1);
    mask.y = T(sign*((int(cub::LaneId()) + 1) & 0x2) >> 1);
}

// Gives the Kepler shuffle index for creating part of the
// brightness matrix.
// Given:
//   thread 0 : pol = I;
//   thread 1 : pol = Q;
//   thread 2 : pol = U;
//   thread 3 : pol = V;
// we want to obtain the following brightness matrix:
//   thread 0 : B = I+Q;
//   thread 1 : B = U+iV;
//   thread 2 : B = U-iV;
//   thread 1 : B = I-Q;
// This gives the indices of [Q,V,V,Q], [1,3,3,1], offset by the warp lane
// Subtracting 1 from these indices gives [I,U,U,I], [0,2,2,0]
__device__ __forceinline__ int brightness_pol_2_shfl_idx()
{
    int vis_idx = (cub::LaneId() >> 2) << 2;
    return ((int(cub::LaneId()) + 1) & 0x2) + vis_idx + 1;
}

// Given the polarisation, generate the brightness matrix
// Assumes that the polarisations I,Q,U,V are present in the
// the pol variable of a group of four threads. i.e.
//   thread 0 : pol = I;
//   thread 1 : pol = Q;
//   thread 2 : pol = U;
//   thread 3 : pol = V;
// Kepler register shuffling is used to retrieve these
// polarisations from each thread to construct the brightness
// value
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__ void create_brightness(
    typename Tr::ct & result,
    const typename Tr::ft & pol)
{
    // Overwrite the result with the mask
    create_brightness_mask<T>(result);
    int shfl_idx = brightness_pol_2_shfl_idx();
    // Get the second polarisation value and multiply it with the mask
    T second_pol = cub::ShuffleIndex(pol, shfl_idx);
    result.x *= second_pol;
    result.y *= second_pol;
    // Add the first polarisation to the real component
    result.x += cub::ShuffleIndex(pol, shfl_idx-1);
}


// Computes the square root of the brightness matrix
// in place. The brightness matrix is assumed to be
// stored in 4 consecutive thread lanes in the pol
// variable.
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__
void create_brightness_sqrt(
    typename Tr::ct & brightness,
    const typename Tr::ft & pol)
{
    // Create the brightness matrix
    create_brightness<T>(brightness, pol);

    // This gives us 0,0,0,0,4,4,4,4,8,8,8,8,...
    int shfl_idx = (cub::LaneId() >> 2) << 2;

    // det = I^2 - Q^2 - U^2 - V^2
    typename Tr::ft I = cub::ShuffleIndex(pol, shfl_idx);
    typename Tr::ft trace = 2*I;
    typename Tr::ft det = I*I;

    typename Tr::ft Q = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= Q*Q;

    typename Tr::ft U = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= U*U;

    typename Tr::ft V = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= V*V;

    // Assumption here is that the determinant
    // of the brightness matrix is positive
    // I^2 - Q^2 - U^2 - V^2 > 0
    typename Tr::ft s = Po::sqrt(det);

    // This gives us 2 0 0 2 2 0 0 2 2 0 0 2
    bool is_diag = ((int(cub::LaneId()) - 1) & 0x2) != 0;
    // Only add s if we're in a lane corresponding to
    // a diagonal matrix entry.
    if(is_diag)
        { brightness.x += s; }

    // Assumption here, trace and 2*s are positive
    // real numbers. 2*s is positive from the
    // assumption of a positive determinant.
    // trace = 2*I > 0 follows from I being positive
    // and real. (although apparently negative I's
    // are positive: see Sunyaev-Zel'dovich effect).
    typename Tr::ft t = Po::sqrt(trace + 2*s);

    // If both s and t are 0, this matrix does
    // not have a square root. But then
    // I,Q,U and V must all be zero. Set
    if(t != T(0.0)) {
        brightness.x /= t;
        brightness.y /= t;
    }
}

} // namespace montblanc

#endif // _MONTBLANC_BRIGHTNESS_CUH