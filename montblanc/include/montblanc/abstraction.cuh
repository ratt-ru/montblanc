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

#ifndef _MONTBLANC_KERNEL_TRAITS_CUH
#define _MONTBLANC_KERNEL_TRAITS_CUH

// CUDA include required for CUDART_PI_F and CUDART_PI
#include <math_constants.h>
// Include cub
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
    // float and complex types
	typedef float FT;
	typedef float2 CT;

    typedef struct __align__(8) {
        float2 XX, XY, YX, YY;
    } visibility_type;

    // Input array types
    typedef float2 lm_type;
    typedef float3 uvw_type;
    typedef float frequency_type;
    typedef float2 complex_phase_type;

    typedef float4 stokes_type;
    typedef float alpha_type;

    typedef float2 point_error_type;
    typedef float2 antenna_scale_type;

    typedef float gauss_param_type;
    typedef float gauss_shape_type;

    typedef float sersic_param_type;
    typedef float sersic_shape_type;

    typedef int32_t antenna_type;
    typedef float2 ant_jones_type;
    typedef int8_t sgn_brightness_type;
    typedef uint8_t flag_type;
    typedef float weight_type;
    typedef float2 die_type;
    typedef float2 vis_type;

public:
	const static bool is_implemented = true;
	constexpr static float cuda_pi = CUDART_PI_F;
};

template <> class kernel_policies<float>
{
public:
	typedef kernel_traits<float> Tr;
    typedef kernel_policies<float> Po;

	__device__ __forceinline__ static
	Tr::CT make_ct(const Tr::FT & real, const Tr::FT & imag)
		{ return ::make_float2(real, imag); }

	__device__ __forceinline__ static
	Tr::FT sqrt(const Tr::FT & value)
		{ return ::sqrtf(value); }

    __device__ __forceinline__ static
    Tr::CT sqrt(const Tr::CT & value)
    {
        constexpr typename Tr::FT half = 0.5;

        typename Tr::FT r = Po::abs(value);
        return Po::make_ct(
            Po::sqrt((r + value.x)*half),
            Po::copysign(
                Po::sqrt((r - value.x)*half), value.y));
    }

	__device__ __forceinline__ static
	Tr::FT min(const Tr::FT & lhs, const Tr::FT & rhs)
		{ return ::fminf(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::FT max(const Tr::FT & lhs, const Tr::FT & rhs)
		{ return ::fmaxf(lhs, rhs); }

    __device__ __forceinline__ static
    Tr::FT clamp(const Tr::FT & value, const Tr::FT & min, const Tr::FT & max)
        { return Po::min(max, Po::max(value, min)); }

    __device__ __forceinline__ static
    Tr::FT floor(const Tr::FT & value)
        { return ::floorf(value); }

	__device__ __forceinline__ static
	Tr::FT pow(const Tr::FT & value, const Tr::FT & exponent)
		{ return ::powf(value, exponent); }

	__device__ __forceinline__ static
	Tr::FT exp(const Tr::FT & value)
		{ return ::expf(value); }

	__device__ __forceinline__ static
	Tr::FT sin(const Tr::FT & value)
		{ return ::sinf(value); }

	__device__ __forceinline__ static
	Tr::FT cos(const Tr::FT & value)
		{ return ::cosf(value); }

	__device__ __forceinline__ static
	void sincos(const Tr::FT & value, Tr::FT * sinptr, Tr::FT * cosptr)
		{ ::sincosf(value, sinptr, cosptr); }

    __device__ __forceinline__ static
    Tr::FT atan2(const Tr::FT & y, const Tr::FT & x)
        { return ::atan2f(y, x); }

    __device__ __forceinline__ static
    Tr::FT arg(const Tr::CT & value)
        { return Po::atan2(value.y, value.x); }

    __device__ __forceinline__ static
    Tr::FT arg(const Tr::FT & value)
        { return Po::atan2(0.0f, value); }

    __device__ __forceinline__ static
    Tr::CT conj(const Tr::CT & value)
        { return Po::make_ct(value.x, -value.y); }

    __device__ __forceinline__ static
    Tr::FT abs_squared(const Tr::CT & value)
        { return value.x*value.x + value.y*value.y; }

    __device__ __forceinline__ static
    Tr::FT abs(const Tr::CT & value)
        { return Po::sqrt(Po::abs_squared(value)); }

    __device__ __forceinline__ static
    Tr::FT abs(const Tr::FT & value)
        { return ::fabsf(value); }

    __device__ __forceinline__ static
    Tr::FT round(const Tr::FT & value)
        { return ::roundf(value); }

    __device__ __forceinline__ static
    Tr::FT rint(const Tr::FT & value)
        { return ::rintf(value); }

    __device__ __forceinline__ static
    Tr::FT rsqrt(const Tr::FT & value)
        { return ::rsqrtf(value); }

    __device__ __forceinline__ static
    Tr::FT copysign(const Tr::FT & magnitude, const Tr::FT & sign)
        { return ::copysignf(magnitude, sign); }
};

template <>
class kernel_traits<double>
{
public:
    // float and complex types
	typedef double FT;
	typedef double2 CT;

    typedef struct __align__(16) {
        double2 XX, XY, YX, YY;
    } visibility_type;

    // Input array types
    typedef double2 lm_type;
    typedef double3 uvw_type;
    typedef double frequency_type;
    typedef double2 complex_phase_type;

    typedef double4 stokes_type;
    typedef double alpha_type;

    typedef double2 point_error_type;
    typedef double2 antenna_scale_type;

    typedef double gauss_param_type;
    typedef double gauss_shape_type;

    typedef double sersic_param_type;
    typedef double sersic_shape_type;

    typedef int32_t antenna_type;
    typedef double2 ant_jones_type;
    typedef uint8_t flag_type;
    typedef int8_t sgn_brightness_type;
    typedef double weight_type;
    typedef double2 die_type;
    typedef double2 vis_type;


public:
	const static bool is_implemented = true;
	constexpr static double cuda_pi = CUDART_PI;
};

template <> class kernel_policies<double>
{
public:
	typedef kernel_traits<double> Tr;
    typedef kernel_policies<double> Po;

	__device__ __forceinline__ static
	Tr::CT make_ct(const Tr::FT & real, const Tr::FT & imag)
		{ return ::make_double2(real, imag); }

	__device__ __forceinline__ static
	Tr::FT sqrt(const Tr::FT & value)
		{ return ::sqrt(value); }

    __device__ __forceinline__ static
    Tr::CT sqrt(const Tr::CT & value)
    {
        constexpr typename Tr::FT half = 0.5;

        typename Tr::FT r = Po::abs(value);
        return Po::make_ct(
            Po::sqrt((r + value.x)*half),
            Po::copysign(
                Po::sqrt((r - value.x)*half), value.y));
    }

	__device__ __forceinline__ static
	Tr::FT min(const Tr::FT & lhs, const Tr::FT & rhs)
		{ return ::fmin(lhs, rhs); }

	__device__ __forceinline__ static
	Tr::FT max(const Tr::FT & lhs, const Tr::FT & rhs)
		{ return ::fmax(lhs, rhs); }

    __device__ __forceinline__ static
    Tr::FT clamp(const Tr::FT & value, const Tr::FT & min, const Tr::FT & max)
        { return Po::min(max, Po::max(value, min)); }

    __device__ __forceinline__ static
    Tr::FT floor(const Tr::FT & value)
        { return ::floor(value); }

	__device__ __forceinline__ static
	Tr::FT pow(const Tr::FT & value, const Tr::FT & exponent)
		{ return ::pow(value, exponent); }

	__device__ __forceinline__ static
	Tr::FT exp(const Tr::FT & value)
		{ return ::exp(value); }

	__device__ __forceinline__ static
	Tr::FT sin(const Tr::FT & value)
		{ return ::sin(value); }

	__device__ __forceinline__ static
	Tr::FT cos(const Tr::FT & value)
		{ return ::cos(value); }

	__device__ __forceinline__ static
	void sincos(const Tr::FT & value, Tr::FT * sinptr, Tr::FT * cosptr)
		{ ::sincos(value, sinptr, cosptr); }

    __device__ __forceinline__ static
    Tr::FT atan2(const Tr::FT & y, const Tr::FT & x)
        { return ::atan2(y, x); }

    __device__ __forceinline__ static
    Tr::FT arg(const Tr::CT & value)
        { return Po::atan2(value.y, value.x); }

    __device__ __forceinline__ static
    Tr::FT arg(const Tr::FT & value)
        { return Po::atan2(0.0, value); }

    __device__ __forceinline__ static
    Tr::CT conj(const Tr::CT & value)
        { return Po::make_ct(value.x, -value.y); }

    __device__ __forceinline__ static
    Tr::FT abs_squared(const Tr::CT & value)
        { return value.x*value.x + value.y*value.y; }

    __device__ __forceinline__ static
    Tr::FT abs(const Tr::CT & value)
        { return Po::sqrt(Po::abs_squared(value)); }

    __device__ __forceinline__ static
    Tr::FT abs(const Tr::FT & value)
        { return ::abs(value); }

    __device__ __forceinline__ static
    Tr::FT round(const Tr::FT & value)
        { return ::round(value); }

    __device__ __forceinline__ static
    Tr::FT rint(const Tr::FT & value)
        { return ::rint(value); }

    __device__ __forceinline__ static
    Tr::FT rsqrt(const Tr::FT & value)
        { return ::rsqrt(value); }

    __device__ __forceinline__ static
    Tr::FT copysign(const Tr::FT & magnitude, const Tr::FT & sign)
        { return ::copysign(magnitude, sign); }
};

dim3 shrink_small_dims(dim3 && block, int X, int Y, int Z);
dim3 grid_from_thread_block(const dim3 & block, int X, int Y, int Z);

// result = rhs*lhs;
template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_multiply(
    typename Tr::CT & result,
    const typename Tr::CT & lhs,
    const typename Tr::CT & rhs)
{
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // a = lhs.x b=lhs.y c=rhs.x d = rhs.y
    result.x = lhs.x*rhs.x - lhs.y*rhs.y;
    result.y = lhs.x*rhs.y + lhs.y*rhs.x;
}

// rhs *= lhs;
template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_conjugate_multiply(
    typename Tr::CT & result,
    const typename Tr::CT & lhs,
    const typename Tr::CT & rhs)
{
    // (a+bi)(c-di) = (ac+bd) + (-ad+bc)i
    // a = lhs.x b=lhs.y c=rhs.x d = rhs.y
    result.x = lhs.x*rhs.x + lhs.y*rhs.y;
    result.y = -lhs.x*rhs.y + lhs.y*rhs.x;
}


// result = rhs*conj(lhs);
template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_multiply_in_place(
    typename Tr::CT & lhs,
    const typename Tr::CT & rhs)
{
    typename Tr::FT tmp = lhs.x;

    lhs.x *= rhs.x;
    lhs.x -= lhs.y*rhs.y;
    lhs.y *= rhs.x;
    lhs.y += tmp*rhs.y;
}

// rhs *= conj(lhs);
template <
    typename T,
    typename Tr=kernel_traits<T>,
    typename Po=kernel_policies<T> >
__device__ __forceinline__ void complex_conjugate_multiply_in_place(
    typename Tr::CT & lhs,
    const typename Tr::CT & rhs)
{
    typename Tr::FT tmp = -lhs.x;

    lhs.x *= rhs.x;
    lhs.x += lhs.y*rhs.y;
    lhs.y *= rhs.x;
    lhs.y += tmp*rhs.y;
}



} // namespace montblanc


#endif // _MONTBLANC_KERNEL_TRAITS_CUH
