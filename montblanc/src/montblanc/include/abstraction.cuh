#ifndef _MONTBLANC_KERNEL_TRAITS_H
#define _MONTBLANC_KERNEL_TRAITS_H

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

} // namespace montblanc

#endif // _MONTBLANC_KERNEL_TRAITS_H