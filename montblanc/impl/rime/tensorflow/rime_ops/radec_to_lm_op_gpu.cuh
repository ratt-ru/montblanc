#if GOOGLE_CUDA

#ifndef RIME_RADEC_TO_LM_OP_GPU_CUH
#define RIME_RADEC_TO_LM_OP_GPU_CUH

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "radec_to_lm_op.h"
#include <montblanc/abstraction.cuh>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN

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
};

// Specialise for double
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};



// CUDA kernel outline
template <typename Traits>
__global__ void rime_radec_to_lm(
    const typename Traits::radec_type * in_radec,
    typename Traits::lm_type * out_lm,
    typename Traits::FT phase_centre_ra,
    typename Traits::FT sin_d0,
    typename Traits::FT cos_d0)

{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using FT = typename Traits::FT;
    using Po = typename montblanc::kernel_policies<FT>;

    using LTr = LaunchTraits<FT>;

    int s = blockIdx.x*blockDim.x + threadIdx.x;

    if(s >= LTr::BLOCKDIMX)
        { return; }

    auto src_radec = in_radec[s];

    // Sine+Cosine of (source RA - phase centre RA)
    FT sin_da, cos_da;
    Po::sincos(src_radec.x - phase_centre_ra, &sin_da, &cos_da);

    // Sine+Cosine of source DEC
    FT sin_d, cos_d;
    Po::sincos(src_radec.y, &sin_d, &cos_d);

    typename Traits::lm_type lm;

    lm.x = cos_d*sin_da;
    lm.y = sin_d*cos_d0 - cos_d*sin_d0*cos_da;

            // // Sin and cosine of (source RA - phase centre RA)
            // auto da = radec(src, 0) - phase_centre(0);
            // auto sin_da = sin(da);
            // auto cos_da = cos(da);

            // // Sine and cosine of source DEC
            // auto sin_d =  sin(radec(src, 1));
            // auto cos_d =  cos(radec(src, 1));

            // lm(src, 0) = cos_d*sin_da;
            // lm(src, 1) = sin_d*cos_d0 - cos_d*sin_d0*cos_da;


    // Set shared buffer to thread index
    out_lm[s] = lm;
}

// Specialise the RadecToLm op for GPUs
template <typename FT>
class RadecToLm<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit RadecToLm(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_radec = context->input(0);
        const auto & in_phase_centre = context->input(1);

        int nsrc = in_radec.dim_size(0);

        // Allocate output tensors
        // Allocate space for output tensor 'lm'
        tf::Tensor * lm_ptr = nullptr;
        tf::TensorShape lm_shape = tf::TensorShape({ nsrc, 2 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, lm_shape, &lm_ptr));


        using LTr = LaunchTraits<FT>;
        using Tr = montblanc::kernel_traits<FT>;

        // Set up our CUDA thread block and grid
        // Set up our kernel dimensions
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            nsrc, 1, 1);
        dim3 grid(montblanc::grid_from_thread_block(
            block, nsrc, 1, 1));

        // Get pointers to flattened tensor data buffers
        const auto fin_radec = reinterpret_cast<
            const typename Tr::radec_type *>(
                in_radec.flat<FT>().data());

        auto fout_lm = reinterpret_cast<
            typename Tr::lm_type *>(
                lm_ptr->flat<FT>().data());

        const auto phase_centre = in_phase_centre.tensor<FT, 1>();

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_radec_to_lm CUDA kernel
        rime_radec_to_lm<Tr>
            <<<grid, block, 0, device.stream()>>>(
                fin_radec, fout_lm,
                phase_centre(0),
                sin(phase_centre(1)),
                cos(phase_centre(1)));

    }
};

MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_RADEC_TO_LM_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
