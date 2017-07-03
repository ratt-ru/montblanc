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

#include <montblanc/abstraction.cuh>

namespace montblanc {

// The base visibility index. 0 0 0 0 4 4 4 4 8 8 8 8
#define _MONTBLANC_VIS_BASE_IDX int(cub::LaneId() & 28)
// Odd polarisation? 0 1 0 1 0 1 0 1 0 1 0 1
#define _MONTBLANC_IS_ODD_POL int(cub::LaneId() & 0x1)
// Even polarisation? 1 0 1 0 1 0 1 0 1 0 1 0
#define _MONTBLANC_IS_EVEN_POL int(_MONTBLANC_IS_ODD_POL == 0)

// | J0   J1 |     | K0   K1 |        | J0.K0+J1.K2     J0.K1+J1.K3 |
// |         |  *  |         |    =   |                             |
// | J2   J3 |     | K2   K3 |        | J2.K0+J3.K2     J2.K1+J3.K3 |
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__
void jones_multiply_4x4_in_place(
    typename Tr::CT & J,
    const typename Tr::CT & K)
{
    // This will produce indexes with the following pattern
    // 1 2 1 2 5 6 5 6 9 10 9 10 13 14 13 14
    int shfl_idx = _MONTBLANC_VIS_BASE_IDX + 1 + _MONTBLANC_IS_ODD_POL;
    // Load in the value to multiply.
    typename Tr::CT shfl_K = cub::ShuffleIndex(K, shfl_idx);

    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = J.x, b=J.y, c=shfl_K.x, d = shfl_K.y
    typename Tr::CT sum;
    sum.x = J.x*shfl_K.x - J.y*shfl_K.y,
    sum.y = J.x*shfl_K.y + J.y*shfl_K.x;

    // Now shuffle the sums
    // This will produce indexes with the following pattern
    // 1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14
    shfl_idx = cub::LaneId() + 1 + -2*_MONTBLANC_IS_ODD_POL;
    sum = cub::ShuffleIndex(sum, shfl_idx);

    // This will produce indexes with the following pattern
    // 0 3 0 3 4 7 4 7 8 11 8 11 12 15 12 15
    shfl_idx = _MONTBLANC_VIS_BASE_IDX + 3*_MONTBLANC_IS_ODD_POL;
    // Load in the polarisation to multiply.
    shfl_K = cub::ShuffleIndex(K, shfl_idx);
    sum.x += J.x*shfl_K.x - J.y*shfl_K.y;
    sum.y += J.x*shfl_K.y + J.y*shfl_K.x;

    J = sum;
}

// | J0   J1 |     | K0   K1 |^H      | J0.K0^H + J1.K1^H   J0.K2^H + J1.K3^H |
// |         |  *  |         |    =   |                                       |
// | J2   J3 |     | K2   K3 |        | J2.K0^H + J3.K1^H   J2.K2^H + J3.K3^H |
template <
    typename T,
    typename Tr=montblanc::kernel_traits<T>,
    typename Po=montblanc::kernel_policies<T> >
__device__ __forceinline__
void jones_multiply_4x4_hermitian_transpose_in_place(
    typename Tr::CT & J,
    const typename Tr::CT & K)
{
    // This will produce indexes with the following pattern
    // 2 1 2 1 6 5 6 5 10 9 10 9 14 13 14 13
    int shfl_idx = _MONTBLANC_VIS_BASE_IDX + 1 + _MONTBLANC_IS_EVEN_POL;
    // Load in the value to multiply.
    typename Tr::CT shfl_K = cub::ShuffleIndex(K, shfl_idx);

    // (a+bi)*conj(c+di) = (a+bi)*(c-di) = (ac+bd) + (-ad+bc)i
    // a = J.x, b=J.y, c=shfl_K.x, d = shfl_K.y
    typename Tr::CT sum;
    sum.x =  J.x*shfl_K.x + J.y*shfl_K.y;
    sum.y = -J.x*shfl_K.y + J.y*shfl_K.x;

    // Now shuffle the sums
    // This will produce indexes with the following pattern
    // 1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14
    shfl_idx = cub::LaneId() + 1 + -2*_MONTBLANC_IS_ODD_POL;
    sum = cub::ShuffleIndex(sum, shfl_idx);

    // This will produce indexes with the following pattern
    // 0 3 0 3 4 7 4 7 8 11 8 11 12 15 12 15
    shfl_idx = _MONTBLANC_VIS_BASE_IDX + 3*_MONTBLANC_IS_ODD_POL;

    // Load in the polarisation to multiply.
    shfl_K = cub::ShuffleIndex(K, shfl_idx);
    // (a+bi)*conj(c+di) = (a+bi)*(c-di) = (ac+bd) + (-ad+bc)i
    sum.x +=  J.x*shfl_K.x + J.y*shfl_K.y;
    sum.y += -J.x*shfl_K.y + J.y*shfl_K.x;

    J = sum;
}

#undef _MONTBLANC_VIS_BASE_IDX
#undef _MONTBLANC_IS_ODD_POL
#undef _MONTBLANC_IS_EVEN_POL

} // namespace montblanc

#endif // _MONTBLANC_JONES_CUH