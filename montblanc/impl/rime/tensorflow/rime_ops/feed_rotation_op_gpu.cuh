#if GOOGLE_CUDA

#ifndef RIME_FEED_ROTATION_OP_GPU_CUH
#define RIME_FEED_ROTATION_OP_GPU_CUH

#include "feed_rotation_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN

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
template <> struct LaunchTraits<double, tensorflow::complex128>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};


// CUDA kernel outline
template <typename Traits>
__global__ void rime_feed_rotation_linear(
    const typename Traits::FT * in_pa_sin,
    const typename Traits::FT * in_pa_cos,
    typename Traits::visibility_type * out_feed_rotation,
    int npa)

{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using VT = typename Traits::visibility_type;

    using Po = typename montblanc::kernel_policies<FT>;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= npa)
        { return; }

    FT pa_sin = in_pa_sin[i];
    FT pa_cos = in_pa_cos[i];

    typename Traits::visibility_type matrix;

    matrix.XX = Po::make_ct(pa_cos, 0);
    matrix.XY = Po::make_ct(pa_sin, 0);
    matrix.YX = Po::make_ct(-pa_sin, 0);
    matrix.YY = Po::make_ct(pa_cos, 0);

    // Set shared buffer to thread index
    out_feed_rotation[i] = matrix;
}

// CUDA kernel outline
template <typename Traits>
__global__ void rime_feed_rotation_circular(
    const typename Traits::FT * in_pa_sin,
    const typename Traits::FT * in_pa_cos,
    typename Traits::visibility_type * out_feed_rotation,
    int npa)

{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using VT = typename Traits::visibility_type;

    using Po = typename montblanc::kernel_policies<FT>;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= npa)
        { return; }

    FT pa_sin = in_pa_sin[i];
    FT pa_cos = in_pa_cos[i];

    typename Traits::visibility_type matrix;

    // exp(i*pa) == cos(pa) + i*sin(pa)
    // exp(-i*pa) == cos(pa) - i*sin(pa)
    matrix.XX = Po::make_ct(pa_cos, -pa_sin); // exp(-i*pa)
    matrix.XY = Po::make_ct(0, 0);
    matrix.YX = Po::make_ct(0, 0);
    matrix.YY = Po::make_ct(pa_cos, pa_sin); // exp(i*pa)

    // Set shared buffer to thread index
    out_feed_rotation[i] = matrix;
}


// Specialise the FeedRotation op for GPUs
template <typename FT, typename CT>
class FeedRotation<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string feed_type;

public:
    explicit FeedRotation(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)

    {
        OP_REQUIRES_OK(context, context->GetAttr("feed_type", &feed_type));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_parallactic_angle_sin = context->input(0);
        const auto & in_parallactic_angle_cos = context->input(1);

        int ntime = in_parallactic_angle_sin.dim_size(0);
        int na = in_parallactic_angle_sin.dim_size(1);
        int npa = ntime*na;

        // Allocate output tensors
        // Allocate space for output tensor 'feed_rotation'
        tf::Tensor * feed_rotation_ptr = nullptr;
        tf::TensorShape feed_rotation_shape = tf::TensorShape({ ntime, na, 4 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, feed_rotation_shape, &feed_rotation_ptr));


        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT, CT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::block_size(npa, 1, 1));
        dim3 grid(montblanc::grid_from_thread_block(
            block, npa, 1, 1));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Get pointers to flattened tensor data buffers
        const auto fin_parallactic_angle_sin = in_parallactic_angle_sin.flat<FT>().data();
        const auto fin_parallactic_angle_cos = in_parallactic_angle_cos.flat<FT>().data();
        auto fout_feed_rotation = reinterpret_cast<typename Tr::visibility_type *>(
                                        feed_rotation_ptr->flat<CT>().data());

        if(feed_type == "linear") {
            // Call the linear feed rotation CUDA kernel
            rime_feed_rotation_linear<Tr>
                <<<grid, block, 0, device.stream()>>>(
                    fin_parallactic_angle_sin,
                    fin_parallactic_angle_cos,
                    fout_feed_rotation,
                    npa);

        } else if(feed_type == "circular") {
            // Call the circular feed rotation CUDA kernel
            rime_feed_rotation_circular<Tr>
                <<<grid, block, 0, device.stream()>>>(
                    fin_parallactic_angle_sin,
                    fin_parallactic_angle_cos,
                    fout_feed_rotation,
                    npa);
        } else {
            // Induce failure
            OP_REQUIRES_OK(context, tf::Status(tf::errors::InvalidArgument(
                "Invalid feed type '", feed_type, "'. "
                "Must be 'linear' or 'circular'")));
        }
    }
};

MONTBLANC_FEED_ROTATION_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ROTATION_OP_GPU_CUH

#endif // #if GOOGLE_CUDA