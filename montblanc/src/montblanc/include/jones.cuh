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

#ifndef _MONTBLANC_JONES_CUH
#define _MONTBLANC_JONES_CUH

#include <cub/cub/cub.cuh>
#include <montblanc/include/abstraction.cuh>

namespace montblanc {

// | J0   J1 |     | K0   K1 |        | J0.K0+J1.K2     J0.K1+J1.K3 |
// |         |  *  |         |    =   |                             |
// | J2   J3 |     | K2   K3 |        | J2.K0+J3.K2     J2.K1+J3.K3 |
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__
void jones_multiply_4x4_in_place(
    typename Tr::ct & J,
    const typename Tr::ct & K)
{
    #define _MONTBLANC_VIS_BASE_IDX ((cub::LaneId() >> 2) << 2)
    #define _MONTBLANC_IS_ODD (cub::LaneId() & 0x1)

    // This will produce indexes with the following pattern
    // 1 2 1 2 5 6 5 6 9 10 9 10 13 14 13 14
    int shfl_idx = _MONTBLANC_VIS_BASE_IDX + 1 + _MONTBLANC_IS_ODD;
    // Load in the value to multiply.
    typename Tr::ct shfl_K = cub::ShuffleIndex(K, shfl_idx);

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = J.x, b=J.y, c=tmp_rhs.x, d = tmp_rhs.y
    typename Tr::ct sum;
    sum.x = J.x*shfl_K.x - J.y*shfl_K.y,
    sum.y = J.x*shfl_K.y + J.y*shfl_K.x;

    // Now shuffle the sums
    // This will produce indexes with the following pattern
    // 1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14
    shfl_idx = cub::LaneId() + 1 + -2*_MONTBLANC_IS_ODD;
    sum = cub::ShuffleIndex(sum, shfl_idx);

    // This will produce indexes with the following pattern
    // 0 3 0 3 4 7 4 7 8 11 8 11 12 15 12 15
    shfl_idx = _MONTBLANC_VIS_BASE_IDX + 3*_MONTBLANC_IS_ODD;
    // Load in the polarisation to multiply.
    shfl_K = cub::ShuffleIndex(K, shfl_idx);
    sum.x += J.x*shfl_K.x - J.y*shfl_K.y;
    sum.y += J.x*shfl_K.y + J.y*shfl_K.x;

    J = sum;
}

} // namespace montblanc

#endif // _MONTBLANC_JONES_CUH