#if GOOGLE_CUDA

#ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH
#define ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#include "zernike_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#define NPOLY 32

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT> struct LaunchTraits {};

// Specialise for float, tensorflow::complex64
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 4;
    static constexpr int BLOCKDIMZ = 4;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};

// Specialise for double, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{

public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 4;
    static constexpr int BLOCKDIMZ = 4;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};

template<typename FT, typename CT>
__device__ __forceinline__ CT mul_CT_FT(FT floatval, CT complexval)
{
    return {floatval * complexval.x, floatval * complexval.y};
}

template <typename FT>
__device__ __forceinline__ FT factorial(unsigned n)
{
    if(n==0) return 1.;
    FT fac = 1;
    for(int i = 1; i <= n; i++) fac *= i;
    return fac;
}

template <typename FT, typename CT, typename Po>
__device__ __forceinline__ FT pre_fac(int k, int n, int m)
{
    FT numerator = factorial<FT>(n - k);
    if(k % 2 == 1) numerator *= -1;
    FT denominator = factorial<FT>(k) * factorial<FT>((n+m)/2.0 - k) * factorial<FT>((n-m)/2.0 - k);
    return numerator / denominator;
}

template <typename FT, typename CT, typename Po>
__device__ __forceinline__ FT zernike_rad(int m, int n, FT rho)
{
    FT radial_component = 0.0;
    for(int k = 0; k < ((n - m) / 2) + 1; k++)
    {
        radial_component += pre_fac<FT, CT, Po>(k, n, m) * Po::pow(rho, n - 2.0 * k);
    }
    return radial_component;
}

template<typename FT, typename CT, typename Po>
__device__ __forceinline__ FT zernike(int j, FT rho, FT phi)
{
    if(rho >= 1) 
    {
        return 0.;
    }
    // Convert from Noll to regular dual index
    int n = 0;
    j += 1;
    int j1 = j - 1;
    while(j1 > n)
    {
        n += 1;
        j1 -= n;
    }
    int m = ((n%2) + 2 * ((j1 + ((n+1)%2)) / 2));
    if(j % 2 == 1) m *= -1;

    // Get Zernike polynomials
    if(m > 0) return zernike_rad<FT, CT, Po>(m, n, rho) 
        * cos(m * phi);
    if(m < 0) return zernike_rad<FT, CT, Po>(-1 * m, n, rho) 
        * sin(-1 * m * phi);
    return zernike_rad<FT, CT, Po>(0, n, rho);
}



// CUDA kernel outline
template <typename Traits> 
__global__ void zernike_dde_zernike(
    const typename Traits::lm_type * in_coords,
    const typename Traits::CT * in_coeffs,
    const int * in_noll_index,
    const typename Traits::point_error_type * in_pointing_error,
    const typename Traits::antenna_scale_type * in_antenna_scaling,
    const typename Traits::FT * in_parallactic_angle_sin,
    const typename Traits::FT * in_parallactic_angle_cos,
    typename Traits::CT * out_zernike_value,
    const int nsrc, const int ntime, const int na, const int nchan, const int npoly)
    
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using point_error_type = typename Traits::point_error_type;
    using antenna_scale_type = typename Traits::antenna_scale_type;
    using lm_type = typename Traits::lm_type;

    using LTr = LaunchTraits<FT>;
    using Po = typename montblanc::kernel_policies<FT>;

    __shared__ struct 
    {
        CT zernike_coeff[LTr::BLOCKDIMY][LTr::BLOCKDIMX >> 2][NPOLY];
        int zernike_noll_indices[LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2][NPOLY];
        antenna_scale_type antenna_scaling[LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2];
        point_error_type pointing_error[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2];
        FT pa_sin[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];
        FT pa_cos[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];
    } shared;


    int corrchan = blockIdx.x * blockDim.x + threadIdx.x;
    int corr = corrchan & 0x3;
    int chan = corrchan >> 2;
    int ant = blockIdx.y * blockDim.y + threadIdx.y;
    int time = blockIdx.z * blockDim.z + threadIdx.z;

    if(corr >= _ZERNIKE_CORRS || chan >= nchan || ant >= na || time >= ntime) return;
        
    if(threadIdx.z == 0)
    {
        shared.antenna_scaling[threadIdx.y][threadIdx.x >> 2] = in_antenna_scaling[
            (((ant * nchan + chan)))];
        for(int p = 0; p < npoly; p++)
        {
            shared.zernike_coeff[threadIdx.y][threadIdx.x >> 2][p] = in_coeffs[
                ((ant * nchan + chan) * npoly + p) * _ZERNIKE_CORRS + corr];
            shared.zernike_noll_indices[threadIdx.y][threadIdx.x >> 2][p] = in_noll_index[
                ((ant * nchan + chan) * npoly + p) * _ZERNIKE_CORRS + corr];
        }
    }
    if((threadIdx.x & 0x03) == 0)
    {
        shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2]  = in_pointing_error[
                (((time * na + ant) * nchan + chan))];
    }

    if(threadIdx.x == 0){
        shared.pa_sin[threadIdx.z][threadIdx.y] = in_parallactic_angle_sin[time * na + ant];
        shared.pa_cos[threadIdx.z][threadIdx.y] = in_parallactic_angle_cos[time * na + ant];
    }
    __syncthreads();

    FT pa_sin = shared.pa_sin[threadIdx.z][threadIdx.y];
    FT pa_cos = shared.pa_cos[threadIdx.z][threadIdx.y];
    
    for(int src = 0; src < nsrc; src++){
        lm_type lm = in_coords[src];
        
        FT l_tmp = lm.x * pa_cos - lm.y * pa_sin;
        FT m_tmp = lm.x * pa_sin + lm.y * pa_cos;
        lm.x = l_tmp;
        lm.y = m_tmp;
        lm.x += shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2].x; 
        lm.x *= shared.antenna_scaling[threadIdx.y][threadIdx.x>>2].x; 
        lm.y += shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2].y; 
        lm.y *= shared.antenna_scaling[threadIdx.y][threadIdx.x>>2].y; 
        
        FT rho = Po::sqrt(lm.x * lm.x + lm.y * lm.y);
        FT phi = Po::atan2(lm.x, lm.y);
        CT zernike_sum = {0, 0};
        for(int poly = 0; poly < npoly; poly++){
            CT zernike_output = mul_CT_FT<FT, CT>(zernike<FT, CT, Po>(shared.zernike_noll_indices[threadIdx.y][threadIdx.x >>2][poly], rho, phi),
                shared.zernike_coeff[threadIdx.y][threadIdx.x >> 2][poly]); // coeff * zernike
            zernike_sum.x += zernike_output.x;
            zernike_sum.y += zernike_output.y;
            }
        out_zernike_value[(((src * ntime + time) * na + ant) * nchan + chan ) * _ZERNIKE_CORRS + corr] = zernike_sum;
    }


}

// Specialise the Zernike op for GPUs
template <typename FT, typename CT>
class Zernike<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit Zernike(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const tf::Tensor & in_coords = context->input(0);
        const tf::Tensor & in_coeffs = context->input(1);
        const tf::Tensor & in_noll_index = context->input(2);
        const tf::Tensor & in_pointing_error = context->input(3);
        const tf::Tensor & in_antenna_scaling = context->input(4);
        const tf::Tensor & in_parallactic_angle_sin = context->input(5);
        const tf::Tensor & in_parallactic_angle_cos = context->input(6);
    
        int nsrc = in_coords.dim_size(0);
        int ntime = in_pointing_error.dim_size(0);
        int na = in_coeffs.dim_size(0);
        int nchan = in_coeffs.dim_size(1);
        int npoly = in_coeffs.dim_size(2);
        
        // Allocate output tensors
        // Allocate space for output tensor 'zernike_value'
        tf::Tensor * zernike_value_ptr = nullptr;
        tf::TensorShape zernike_value_shape = tf::TensorShape({ nsrc, ntime, na, nchan, _ZERNIKE_CORRS });
        typedef montblanc::kernel_traits<FT> Tr;
        using LTr = LaunchTraits<typename Tr::FT>;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, zernike_value_shape, &zernike_value_ptr));

        OP_REQUIRES(context, npoly <= NPOLY, tf::errors::InvalidArgument("Npoly is too large. Must be %d or less.\n", NPOLY));

        // Set up our CUDA thread block and grid

        dim3 block(LTr::block_size(_ZERNIKE_CORRS * nchan, na, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            block, _ZERNIKE_CORRS * nchan, na, ntime)); 

        // Get pointers to flattened tensor data buffers
        auto coords = reinterpret_cast< //lm_type coords
            const typename Tr::lm_type *>(
                in_coords.flat<FT>().data());
        auto coeffs = reinterpret_cast< //CT coeffs
            const typename Tr::CT *>(
                in_coeffs.flat<CT>().data());
        auto noll_index = in_noll_index.flat<int>().data(); //int noll_index
        auto pointing_error = reinterpret_cast< //point_error_type pointing_error
            const typename Tr::point_error_type *>(
                in_pointing_error.flat<FT>().data());
        auto antenna_scaling = reinterpret_cast< //antenna_scale_type antenna_scaling
            const typename Tr::antenna_scale_type *>(
                in_antenna_scaling.flat<FT>().data());
        auto parallactic_angle_sin = reinterpret_cast< // FT parallactic_angle_sin
            const typename Tr::FT *>(
                in_parallactic_angle_sin.flat<FT>().data());
        auto parallactic_angle_cos = reinterpret_cast< // FT parallactic_angle_cos
            const typename Tr::FT *>(
                in_parallactic_angle_cos.flat<FT>().data());
        auto zernike_value = reinterpret_cast<typename Tr::CT *>(
            zernike_value_ptr->flat<CT>().data());
        
        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the zernike_dde_zernike CUDA kernel
        zernike_dde_zernike<Tr>
            <<<grid, block, 0, device.stream()>>>(
                coords,
                coeffs,
                noll_index,
                pointing_error,
                antenna_scaling,
                parallactic_angle_sin,
                parallactic_angle_cos,
                zernike_value,
                nsrc, ntime, na, nchan, npoly);
        cudaError_t e = cudaGetLastError(); 
        if(e!=cudaSuccess) 
        {                                              
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           
            exit(0); 
        }         
    }
};

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA