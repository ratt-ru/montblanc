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

#ifndef _MONTBLANC_KERNEL_TRAITS_H
#define _MONTBLANC_KERNEL_TRAITS_H

#include <cub/cub/cub.cuh>

namespace montblanc {

template <typename T> class kernel_traits
{
public:
	const static bool is_implemented = false;
};

template <typename T> class kernel_policies
{
public:
	const static bool is_implemented = false;
};

template <> class kernel_traits<float>
{
public:
	typedef float ft;
	typedef float2 ct;

public:
	const static bool is_implemented = true;
	const static float cuda_pi = CUDART_PI_F;
};

template <> class kernel_policies<float>
{
public:
	typedef kernel_traits<float> Tr;

	__device__ __forceinline__ static
	Tr::ct make_ct(const Tr::ft & real, const Tr::ft & imag)
		{ return ::make_float2(real, imag); }

	__device__ __forceinline__ static
	Tr::ft sqrt(const Tr::ft & value)
		{ return ::sqrtf(value); }

	__device__ __forceinline__ static
	Tr::ft min(const Tr::ft & lhs, const Tr::ft & rhs)
		{ return ::fminf(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::ft max(const Tr::ft & lhs, const Tr::ft & rhs)
		{ return ::fmaxf(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::ft pow(const Tr::ft & value, const Tr::ft & exponent)
		{ return ::powf(value, exponent); }

	__device__ __forceinline__ static
	Tr::ft exp(const Tr::ft & value)
		{ return ::expf(value); }

	__device__ __forceinline__ static
	Tr::ft sin(const Tr::ft & value)
		{ return ::sinf(value); }

	__device__ __forceinline__ static
	Tr::ft cos(const Tr::ft & value)
		{ return ::cosf(value); }

	__device__ __forceinline__ static
	void sincos(const Tr::ft & value, Tr::ft * sinptr, Tr::ft * cosptr)
		{ ::sincosf(value, sinptr, cosptr); }
};

template <>
class kernel_traits<double>
{
public:
	typedef double ft;
	typedef double2 ct;

public:
	const static bool is_implemented = true;
	const static double cuda_pi = CUDART_PI;
};

template <> class kernel_policies<double>
{
public:
	typedef kernel_traits<double> Tr;

	__device__ __forceinline__ static
	Tr::ct make_ct(const Tr::ft & real, const Tr::ft & imag)
		{ return ::make_double2(real, imag); }

	__device__ __forceinline__ static
	Tr::ft sqrt(const Tr::ft & value)
		{ return ::sqrt(value); }

	__device__ __forceinline__ static
	Tr::ft min(const Tr::ft & lhs, const Tr::ft & rhs)
		{ return ::fmin(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::ft max(const Tr::ft & lhs, const Tr::ft & rhs)
		{ return ::fmax(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::ft pow(const Tr::ft & value, const Tr::ft & exponent)
		{ return ::pow(value, exponent); }

	__device__ __forceinline__ static
	Tr::ft exp(const Tr::ft & value)
		{ return ::exp(value); }

	__device__ __forceinline__ static
	Tr::ft sin(const Tr::ft & value)
		{ return ::sin(value); }

	__device__ __forceinline__ static
	Tr::ft cos(const Tr::ft & value)
		{ return ::cos(value); }

	__device__ __forceinline__ static
	void sincos(const Tr::ft & value, Tr::ft * sinptr, Tr::ft * cosptr)
		{ ::sincos(value, sinptr, cosptr); }
};

template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_multiply(
    typename Tr::ct & result,
    const typename Tr::ct & lhs,
    const typename Tr::ct & rhs)
{
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = lhs.x b=lhs.y c=rhs.x d = rhs.y
    result.x = lhs.x*rhs.x - lhs.y*rhs.y;
    result.y = lhs.x*rhs.y + lhs.y*rhs.x;
}

template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_multiply_in_place(
    typename Tr::ct & lhs,
    const typename Tr::ct & rhs)
{
    typename Tr::ft tmp = lhs.x;

    lhs.x *= rhs.x;
    lhs.x -= lhs.y*rhs.y;
    lhs.y *= rhs.x;
    lhs.y += tmp*rhs.y;
}

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
    int sign = ((int(threadIdx.x) - 2) & 0x2)  - 1;
    mask.x = T(sign*((int(threadIdx.x) - 1) & 0x2) >> 1);
    mask.y = T(sign*((int(threadIdx.x) + 1) & 0x2) >> 1);
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
__device__ __forceinline__ int brightness_pol_2_shfl_idx(void)
    { return 1 + ((int(threadIdx.x) + 1) & 0x2) + ((int(threadIdx.x) >> 2) << 2); }

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
    T second_pol = cub::ShuffleBroadcast(pol, shfl_idx);
    result.x *= second_pol;
    result.y *= second_pol;
    // Add the first polarisation to the real component
    result.x += cub::ShuffleBroadcast(pol, shfl_idx-1);
}

} // namespace montblanc

#endif // _MONTBLANC_KERNEL_TRAITS_H