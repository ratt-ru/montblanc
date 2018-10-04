#if GOOGLE_CUDA

#ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH
#define ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#include "zernike_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#define NPOLY 16

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
    static constexpr int BLOCKDIMX = 16;
    static constexpr int BLOCKDIMY = 16;
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
    static constexpr int BLOCKDIMX = 16;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 4;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};

template<typename FT, typename CT>
__device__ __forceinline__ CT mul_CT_FT(FT floatval, CT complexval){
    return {floatval * complexval.x, floatval * complexval.y};
}

template <typename FT, typename CT>
__device__ __forceinline__ FT factorial(unsigned n){
    if(n==0) return 1.;
    FT fac = 1;
    for(int i = 1; i <= n; i++) fac *= i;
    return fac;
}

template <typename FT, typename CT>
__device__ __forceinline__ FT pre_fac(int k, int n, int m){
    FT numerator = pow(-1, k) * factorial<FT, CT>(n - k);
    FT denominator = factorial<FT, CT>(k) * factorial<FT, CT>((n+m)/2.0 - k) * factorial<FT, CT>((n-m)/2.0 - k);
    return numerator / denominator;
}

template <typename FT, typename CT>
__device__ __forceinline__ FT zernike_rad(int m, int n, FT rho){
    FT radial_component = 0.0;
    for(int k = 0; k < ((n - m) / 2) + 1; k++){
        radial_component += pre_fac<FT, CT>(k, n, m) * pow(rho, n - 2.0 * k);
    }
    return radial_component;
}

template<typename FT, typename CT>
__device__ __forceinline__ FT zernike(int j, FT rho, FT phi){
    if(rho >= 1) return 0.;

    // Convert from Noll to regular dual index
    int n = 0;
    j += 1;
    int j1 = j - 1;
    while(j1 > n){
        n += 1;
        j1 -= n;
    }
    int m = ((n%2) + 2 * ((j1 + ((n+1)%2)) / 2));
    if(j % 2 == 1) m *= -1;

    // Get Zernike polynomials
    if(m > 0) return zernike_rad<FT, CT>(m, n, rho) 
    * cos(m * phi);
    else if(m < 0) return zernike_rad<FT, CT>(-m, n, rho) 
    * sin(-m * phi);
    return zernike_rad<FT, CT>(0, n, rho);
}



// CUDA kernel outline
template <typename Traits> 
__global__ void zernike_dde_zernike(
    const typename Traits::FT * in_coords,
    const typename Traits::CT * in_coeffs,
    const int * in_noll_index,
    const typename Traits::point_error_type * in_pointing_error,
    const typename Traits::antenna_scale_type * in_antenna_scaling,
    typename Traits::CT * out_zernike_value,
    const int nsrc, const int ntime, const int na, const int nchan, const int npoly)
    
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using point_error_type = typename Traits::point_error_type;
    using antenna_scale_type = typename Traits::antenna_scale_type;

    using LTr = LaunchTraits<FT>;
    using Po = typename montblanc::kernel_policies<FT>;

    __shared__ struct {
        CT zernike_coeff[LTr::BLOCKDIMY][LTr::BLOCKDIMX >> 2][NPOLY];
        int zernike_noll_indices[LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2][NPOLY];
        antenna_scale_type antenna_scaling[LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2];
        point_error_type pointing_error[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][LTr::BLOCKDIMX>>2];
    } shared;


    int corrchan = blockIdx.x * blockDim.x + threadIdx.x;
    int corr = corrchan & 0x3;
    int chan = corrchan >> 2;
    int ant = blockIdx.y * blockDim.y + threadIdx.y;
    int time = blockIdx.z * blockDim.z + threadIdx.z;

  int i = threadIdx.z * 4 + (threadIdx.x & 0x03);

    if(i < npoly){  
        shared.zernike_coeff[threadIdx.y][threadIdx.x >> 2][i] = in_coeffs[((ant * nchan + chan) * npoly + i) * 4 + corr];
        shared.zernike_noll_indices[threadIdx.y][threadIdx.x >> 2][i] = in_noll_index[((ant * nchan + chan) * npoly + i) * 4 + corr];
    }
        
    if(threadIdx.z == 0) shared.antenna_scaling[threadIdx.y][threadIdx.x >> 2] = in_antenna_scaling[(((ant * nchan + chan) * 4 + corr))];
    if((threadIdx.x & 0x03) == 0){
        shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2]  = in_pointing_error[(((time * na + ant) * nchan + chan) * 4 + corr)];

    }

    __syncthreads();

    if(corr >= 4 || chan >= nchan || ant >= na || time >= ntime) return;
    for(int src = 0; src < nsrc; src++){
        FT l = (in_coords[(src * 4 + corr) * 2 ] + shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2].x) * shared.antenna_scaling[threadIdx.y][threadIdx.x>>2].x; 
        FT m = (in_coords[(src * 4 + corr) * 2 + 1] + shared.pointing_error[threadIdx.z][threadIdx.y][threadIdx.x>>2].y) * shared.antenna_scaling[threadIdx.y][threadIdx.x>>2].y; 
        FT rho = Po::sqrt(l * l + m * m);
        FT phi = Po::atan2(l, m);
        CT zernike_sum = {0, 0};
        for(int poly = 0; poly < npoly; poly++){
            CT zernike_output = mul_CT_FT<FT, CT>(zernike<FT, CT>(shared.zernike_noll_indices[threadIdx.y][threadIdx.x >>2][poly], rho, phi), shared.zernike_coeff[threadIdx.y][threadIdx.x >> 2][poly]);
            zernike_sum.x += zernike_output.x;
            zernike_sum.y += zernike_output.y;
        }
        out_zernike_value[(((src * ntime + time) * na + ant) * nchan + chan ) * 4 + corr] = zernike_sum;
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



        int nsrc = in_coords.dim_size(0);
        int ntime = in_pointing_error.dim_size(0);
        int na = in_coeffs.dim_size(0);
        int nchan = in_coeffs.dim_size(1);
        int npoly = in_coeffs.dim_size(2);
        

        // Allocate output tensors
        // Allocate space for output tensor 'zernike_value'
        tf::Tensor * zernike_value_ptr = nullptr;
        tf::TensorShape zernike_value_shape = tf::TensorShape({ nsrc, ntime, na, nchan, 4 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, zernike_value_shape, &zernike_value_ptr));
        typedef montblanc::kernel_traits<FT> Tr;
        using LTr = LaunchTraits<typename Tr::FT>;
        // Set up our CUDA thread block and grid

        dim3 block(LTr::block_size(4 * nchan, na, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            block, 4 * nchan, na, ntime));
        printf("Setting up block(%d, %d, %d) and grid(%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);

        // Get pointers to flattened tensor data buffers
        auto coords = reinterpret_cast< //lm_type coords
            const typename Tr::FT *>(
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
                zernike_value,
                nsrc, ntime, na, nchan, npoly);
        cudaError_t e = cudaGetLastError(); 
        if(e!=cudaSuccess) {                                              
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           
            exit(0); 
            } 
                
    }
};

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA