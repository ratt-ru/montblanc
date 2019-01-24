#if GOOGLE_CUDA

#ifndef RIME_FEED_ANGLE_OP_GPU_CUH
#define RIME_FEED_ANGLE_OP_GPU_CUH

#include "feed_angle_op.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN

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
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};
// Specialise for float, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float, tensorflow::complex128>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};
// Specialise for double, tensorflow::complex64
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double, tensorflow::complex64>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};
// Specialise for double, tensorflow::complex128
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double, tensorflow::complex128>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};


// CUDA kernel outline
template <typename FT, typename CT>
__global__ void rime_feed_angle(
    const FT * in_feed_angle,
    const FT * in_parallactic_angle,
    CT * out_feed_angle_rotation)

{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT, CT>;
    __shared__ int buffer[LTr::BLOCKDIMX];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= LTr::BLOCKDIMX)
        { return; }

    // Set shared buffer to thread index
    buffer[i] = i;
}

// Specialise the FeedAngle op for GPUs
template <typename FT, typename CT>
class FeedAngle<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit FeedAngle(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_feed_angle = context->input(0);
        const auto & in_parallactic_angle = context->input(1);

        int ntime = in_parallactic_angle.dim_size(0);
        int na = in_parallactic_angle.dim_size(1);

        // Allocate output tensors
        // Allocate space for output tensor 'feed_angle_rotation'
        tf::Tensor * feed_angle_rotation_ptr = nullptr;
        tf::TensorShape feed_angle_rotation_shape = tf::TensorShape({
            ntime, na, 4 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, feed_angle_rotation_shape, &feed_angle_rotation_ptr));


        using LTr = LaunchTraits<FT, CT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::BLOCKDIMX);
        dim3 grid(1);

        // Get pointers to flattened tensor data buffers
        const auto fin_feed_angle = in_feed_angle.flat<FT>().data();
        const auto fin_parallactic_angle = in_parallactic_angle.flat<FT>().data();
        auto fout_feed_angle_rotation = feed_angle_rotation_ptr->flat<CT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_feed_angle CUDA kernel
        rime_feed_angle<FT, CT>
            <<<grid, block, 0, device.stream()>>>(
                fin_feed_angle,
                fin_parallactic_angle,
                fout_feed_angle_rotation);

    }
};

MONTBLANC_FEED_ANGLE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ANGLE_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
