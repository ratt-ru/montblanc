#if GOOGLE_CUDA

#ifndef RIME_BRIGHTNESS_OP_GPU_CUH
#define RIME_BRIGHTNESS_OP_GPU_CUH

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "brightness_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/brightness.cuh>


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN

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
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
    }
};

// Specialise for double, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
    }
};


// CUDA kernel outline
template <typename Traits>
__global__ void rime_brightness(
    const typename Traits::FT * in_stokes,
    typename Traits::CT * out_brightness,
    int nrowpols)

{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= nrowpols)
        { return; }

    // Create and set the brightness matrix
    FT polarisation = in_stokes[i];
    CT correlation;
    create_brightness<FT>(correlation, polarisation);
    out_brightness[i] = correlation;
}

// Specialise the Brightness op for GPUs
template <typename FT, typename CT>
class Brightness<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit Brightness(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

   // Create reference to input Tensorflow tensors
        const auto & in_stokes = context->input(0);

        // Allocate output tensors
        // Allocate space for output tensor 'brightness'
        tf::Tensor * brightness_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_stokes.shape(), &brightness_ptr));

        // Extract Eigen tensors
        // auto stokes = in_stokes.flat_inner_dims<FT>();
        // auto brightness = brightness_ptr->flat_inner_dims<CT>();

        // Cast input into CUDA types defined within the Traits class
        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;

        auto flat_stokes = in_stokes.flat_inner_dims<FT>();
        auto nrows = flat_stokes.dimension(0);
        auto npols = flat_stokes.dimension(1);
        auto nrowpols = nrows*npols;

        OP_REQUIRES(context, npols == 4,
            tf::errors::InvalidArgument("Polarisations must be '4'."));

        // Set up our CUDA thread block and grid
        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(nrowpols, 1, 1));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, nrowpols, 1, 1));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Cast to the cuda types expected by the kernel
        auto stokes_gpu = reinterpret_cast<const typename Tr::FT *>(
            flat_stokes.data());
        auto brightness_gpu = reinterpret_cast<typename Tr::CT *>(
            brightness_ptr->flat<CT>().data());

        // Call the rime_brightness CUDA kernel
        rime_brightness<Tr>
            <<<grid, blocks, 0, device.stream()>>>(
                stokes_gpu, brightness_gpu, nrowpols);
    }
};

MONTBLANC_BRIGHTNESS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_BRIGHTNESS_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
