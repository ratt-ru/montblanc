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

#include <montblanc/abstraction.cuh>

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
__device__ __forceinline__
void create_brightness_mask(typename Tr::CT & mask)
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
__device__ __forceinline__
int brightness_pol_2_shfl_idx()
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
__device__ __forceinline__
void create_brightness(
    typename Tr::CT & result,
    const typename Tr::FT & pol)
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
    typename Tr::CT & brightness,
    const typename Tr::FT & pol)
{
    constexpr typename Tr::FT two = 2.0;
    constexpr typename Tr::FT half = 0.5;

    // Create the brightness matrix
    create_brightness<T>(brightness, pol);

    // This gives us 0,0,0,0,4,4,4,4,8,8,8,8,...
    int shfl_idx = (cub::LaneId() >> 2) << 2;

    // det = I^2 - Q^2 - U^2 - V^2
    typename Tr::FT I = cub::ShuffleIndex(pol, shfl_idx);
    typename Tr::FT trace = two*I;
    typename Tr::FT I_squared = I*I;
    typename Tr::FT det = I_squared;

    typename Tr::FT Q = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= Q*Q;

    typename Tr::FT U = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= U*U;

    typename Tr::FT V = cub::ShuffleIndex(pol, ++shfl_idx);
    det -= V*V;

    // This gives us 2 0 0 2 2 0 0 2 2 0 0 2
    bool is_diag = ((int(cub::LaneId()) - 1) & 0x2) != 0;

    // Scalar matrix. Take square root of diagonals and return
    if(det == I_squared)
    {
        if(is_diag)
            { brightness = Po::sqrt(brightness); }

        return;
    }

    // Square root of the determinant
    typename Tr::FT r = std::abs(det);
    typename Tr::CT s = Po::make_ct(
        Po::sqrt((r + det)*half),
        Po::sqrt((r - det)*half));

    // Only add s if we're in a lane corresponding to
    // a diagonal matrix entry.
    if(is_diag)
    {
        brightness.x += s.x;
        brightness.y += s.y;
    }

    s.x *= two; s.y *= two;
    s.x += trace;

    r = Po::abs(s);

    // Prevent nans.
    if(r > 0.0) {
        // Square root of 2*s + trace
        typename Tr::CT t = Po::make_ct(Po::sqrt((r + s.x)*half),
            Po::copysign(Po::sqrt((r - s.x)*half), s.y));

        // We get this automagically
        // r is the magnitude of 2*s + trace, which we take
        // the square root of to obtain t. Hence
        // r = t.x*t.x + t.y*t.y;
        typename Tr::CT b = brightness;
        brightness.x = (b.x*t.x + b.y*t.y)/r;
        brightness.y = (b.y*t.x - b.x*t.y)/r;
    }
}

} // namespace montblanc

#endif // _MONTBLANC_BRIGHTNESS_CUH