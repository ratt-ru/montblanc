#if GOOGLE_CUDA

#ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH
#define ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#include "zernike_op.h"
#include <montblanc/abstraction.cuh>
#include <math.h>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT, typename CT> struct LaunchTraits {};

// Specialise for float, tensorflow::complex64
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float, tensorflow::complex64>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

// Specialise for double, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double, tensorflow::complex128>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

template <typename FT, typename CT>
__device__ __forceinline__ FT factorial(unsigned n){
    if(n==0) return 1;
    FT fac = 1;
    for(int i = 1; i <= n; i++) fac *= 1;
    return fac;
}

template <typename FT, typename CT>
__device__ __forceinline__ FT pre_fac(int k, int n, int m){
    FT numerator = factorial<FT, CT>(n - k);
    if(k % 2 == 1) numerator += -1;
    FT denominator = factorial<FT, CT>(k) * factorial<FT, CT>((n+m)/2.0 - k) * factorial<FT, CT>((n-m)/2.0 - k);
    return numerator / denominator;
}

template <typename FT, typename CT>
__device__ __forceinline__ FT zernike_rad(int m, int n, FT rho){
    if(n < 0 || m < 0 || m > n)return -1;
    FT radial_component = 0.0;
    for(int k = 0; k < ((n - m) / 2) + 1; k++){
        radial_component += pre_fac<FT, CT>(k, n, m) * pow(rho, n - 2.0 * k);
    }
    return radial_component;
}
        
template<typename FT, typename CT>
__device__ __forceinline__ CT zernike(int j, FT rho, FT phi){
    if(rho > 1) 0;

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
template <typename FT, typename CT> 
__global__ void zernike_dde_zernike(
    const FT * in_coords,
    const CT * in_coeffs,
    const FT * in_noll_index,
    CT * out_zernike_value,
    const int nsrc, const int ntime, const int na, const int nchan, const int npoly)
    
{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT, CT>;
    __shared__ int buffer[LTr::BLOCKDIMX];

    //(loop,  z  , y  ,       x     )
    // (src, time, ant, [chan, corr])
    int antchan = blockIdx.x*blockDim.x + threadIdx.x; //combine chan and ant
    int chan = antchan % nchan;
    int ant = antchan / nchan;
    int time = blockIdx.y*blockDim.y + threadIdx.y;
    int src = blockIdx.z*blockDim.z + threadIdx.z;
    
    if(antchan >= LTr::BLOCKDIMX)
        { return; }
    FT l = in_coords[0, src, time, ant, chan];
    FT m = in_coords[1, src, time, ant, chan];
    FT rho = sqrt(l * l + m * m);
    FT phi = atan2(l, m);
    int noll_index = (int)in_noll_index[ant, chan, 0];
    CT zernike_sum = in_coeffs[ant, chan, 0] * zernike<FT, CT>(noll_index, rho, phi);
    for(int poly = 1; poly < npoly; poly++){
        noll_index = in_noll_index[ant, chan, poly];
        zernike_sum = zernike_sum +  in_coeffs[ant, chan, poly] * zernike<FT, CT>(noll_index, rho, phi);
    }
    out_zernike_value[src, time, ant, chan] = zernike_sum;
    
    //out_zernike_value(0,0,0,0) = 0;

    // Set shared buffer to thread index
    buffer[chan] = chan;
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
        const auto & in_coords = context->input(0);
        const auto & in_coeffs = context->input(1);
        const auto & in_noll_index = context->input(2);
        

        const int nsrc = in_coords.dim_size(1);
        const int ntime = in_coords.dim_size(2);
        const int na = in_coords.dim_size(3);
        const int nchan = in_coords.dim_size(4);
        const int npoly = in_coeffs.dim_size(2);

        // Allocate output tensors
        // Allocate space for output tensor 'zernike_value'
        tf::Tensor * zernike_value_ptr = nullptr;
        tf::TensorShape zernike_value_shape = tf::TensorShape({ 
            nsrc, 
            ntime, 
            na, 
            nchan });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, zernike_value_shape, &zernike_value_ptr));
        

        using LTr = LaunchTraits<FT, CT>;

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            nchan, na, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, nsrc, 1,1));

        // Get pointers to flattened tensor data buffers
        const auto fin_coords = in_coords.flat<FT>().data();
        const auto fin_coeffs = in_coeffs.flat<CT>().data();
        const auto fin_noll_index = in_noll_index.flat<FT>().data();
        auto fout_zernike_value = zernike_value_ptr->flat<CT>().data();
        

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the zernike_dde_zernike CUDA kernel
        zernike_dde_zernike<FT, CT>
            <<<grid, block, 0, device.stream()>>>(
                fin_coords,
                fin_coeffs,
                fin_noll_index,
                fout_zernike_value,
                nsrc, ntime, na, nchan, npoly);
                
    }
};

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA