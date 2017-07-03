#if GOOGLE_CUDA

#ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH
#define RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH

#include "parallactic_angle_sin_cos_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT> struct LaunchTraits {};

// Specialise for float
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float>
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
// Specialise for double
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
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
template <typename FT>
__global__ void rime_parallactic_angle_sin_cos(
    const FT * in_parallactic_angle,
    FT * out_pa_sin,
    FT * out_pa_cos,
    int npa)

{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT>;
    using Po = typename montblanc::kernel_policies<FT>;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= npa)
        { return; }

    // Set shared buffer to thread index
    FT sin_pa;
    FT cos_pa;

    Po::sincos(in_parallactic_angle[i], &sin_pa, &cos_pa);

    out_pa_sin[i] = sin_pa;
    out_pa_cos[i] = cos_pa;
}

// Specialise the ParallacticAngleSinCos op for GPUs
template <typename FT>
class ParallacticAngleSinCos<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ParallacticAngleSinCos(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_parallactic_angle = context->input(0);

        int ntime = in_parallactic_angle.dim_size(0);
        int na = in_parallactic_angle.dim_size(1);
        int npa = ntime*na;

        // Allocate output tensors
        // Allocate space for output tensor 'pa_sin'
        tf::Tensor * pa_sin_ptr = nullptr;
        tf::TensorShape pa_sin_shape = tf::TensorShape({ ntime, na });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, pa_sin_shape, &pa_sin_ptr));
        // Allocate space for output tensor 'pa_cos'
        tf::Tensor * pa_cos_ptr = nullptr;
        tf::TensorShape pa_cos_shape = tf::TensorShape({ ntime, na });
        OP_REQUIRES_OK(context, context->allocate_output(
            1, pa_cos_shape, &pa_cos_ptr));


        using LTr = LaunchTraits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::block_size(npa, 1, 1));
        dim3 grid(montblanc::grid_from_thread_block(
            block, npa, 1, 1));

        // Get pointers to flattened tensor data buffers
        const auto fin_parallactic_angle = in_parallactic_angle.flat<FT>().data();
        auto fout_pa_sin = pa_sin_ptr->flat<FT>().data();
        auto fout_pa_cos = pa_cos_ptr->flat<FT>().data();


        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_parallactic_angle_sin_cos CUDA kernel
        rime_parallactic_angle_sin_cos<FT>
            <<<grid, block, 0, device.stream()>>>(
                fin_parallactic_angle,
                fout_pa_sin,
                fout_pa_cos,
                npa);

    }
};

MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH

#endif // #if GOOGLE_CUDA