#ifndef RIME_PHASE_OP_GPU_H_
#define RIME_PHASE_OP_GPU_H_

#if GOOGLE_CUDA

#include "constants.h"
#include "phase_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace phase {

// Traits class defined by float and complex types
template <typename FT> class LaunchTraits;

// Specialise for float
template <> class LaunchTraits<float>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int narow)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, narow, 1);
    }
};

// Specialise for double
template <> class LaunchTraits<double>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 4;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int narow)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, narow, 1);
    }
};

typedef Eigen::GpuDevice GPUDevice;

// CUDA kernel computing the phase term
template <typename Traits>
__global__ void rime_phase(
    const typename Traits::lm_type * lm,
    const typename Traits::uvw_type * uvw,
    const typename Traits::frequency_type * frequency,
    typename Traits::complex_phase_type * complex_phase,
    int nsrc, int narow, int nchan)
{
    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int arow = blockIdx.y*blockDim.y + threadIdx.y;

    if(chan >= nchan || arow >= narow)
        { return; }

    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;

    using Po = typename montblanc::kernel_policies<FT>;
    using LTr = typename montblanc::phase::LaunchTraits<FT>;

    constexpr FT one = FT(1.0);

    // Lightspeed
    constexpr FT lightspeed = 299792458;
    constexpr FT two_pi_over_c = FT(-2.0*M_PI/lightspeed);

    __shared__ struct {
        typename Traits::uvw_type uvw[LTr::BLOCKDIMY];
        typename Traits::frequency_type freq[LTr::BLOCKDIMX];
    } shared;

    // UVW coordinates vary by antenna row, but not channel
    if(threadIdx.x == 0)
        { shared.uvw[threadIdx.y] = uvw[arow]; }

    // Wavelengths vary by channel, not antenna row
    if(threadIdx.y == 0)
        { shared.freq[threadIdx.x] = frequency[chan]; }

    __syncthreads();

    // Iterate over sources
    for(int src=0; src < nsrc; ++src)
    {
        // Calculate the n coordinate
        typename Traits::lm_type r_lm = lm[src];
        FT n = Po::sqrt(one - r_lm.x*r_lm.x - r_lm.y*r_lm.y) - one;

        // Calculate the real phase term
        FT real_phase = shared.uvw[threadIdx.y].z*n +
                        shared.uvw[threadIdx.y].y*r_lm.y +
                        shared.uvw[threadIdx.y].x*r_lm.x;

        real_phase *= two_pi_over_c*shared.freq[threadIdx.x];

        CT cplx_phase;
        Po::sincos(real_phase, &cplx_phase.y, &cplx_phase.x);

        int i = (src*narow + arow)*nchan + chan;
        complex_phase[i] = cplx_phase;
    }
}

// Partially specialise Phase for GPUDevice
template <typename FT, typename CT>
class Phase<GPUDevice, FT, CT> : public tensorflow::OpKernel {
public:
    explicit Phase(tensorflow::OpKernelConstruction * context) : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_uvw = context->input(1);
        const tf::Tensor & in_frequency = context->input(2);

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int narow = in_uvw.dim_size(0);
        int nchan = in_frequency.dim_size(0);

        // Reason about our output shape
        tf::TensorShape complex_phase_shape({nsrc, narow, nchan});

        // Create a pointer for the complex_phase result
        tf::Tensor * complex_phase_ptr = nullptr;

        // Allocate memory for the complex_phase
        OP_REQUIRES_OK(context, context->allocate_output(
            0, complex_phase_shape, &complex_phase_ptr));

        if (complex_phase_ptr->NumElements() == 0)
            { return; }

        // Cast input into CUDA types defined within the Traits class
        typedef montblanc::kernel_traits<FT> Tr;
        typedef typename montblanc::phase::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(nchan, narow));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, nchan, narow, 1));

        // Cast to the cuda types expected by the kernel
        auto lm = reinterpret_cast<const typename Tr::lm_type *>(
            in_lm.flat<FT>().data());
        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto complex_phase = reinterpret_cast<
            typename Tr::complex_phase_type *>(
                complex_phase_ptr->flat<CT>().data());

        // Get the stream
        const auto & stream = context->eigen_device<GPUDevice>().stream();

        // Invoke the kernel
        rime_phase<Tr> <<<grid, blocks, 0, stream>>>(
            lm, uvw, frequency, complex_phase,
            nsrc, narow, nchan);
    }
};

} // namespace phase {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #define RIME_PHASE_OP_GPU_H
