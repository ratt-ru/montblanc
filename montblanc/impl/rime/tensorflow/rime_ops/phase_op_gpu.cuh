#ifndef RIME_PHASE_OP_GPU_H_
#define RIME_PHASE_OP_GPU_H_

#if GOOGLE_CUDA

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
    static constexpr int BLOCKDIMY = 8;
    static constexpr int BLOCKDIMZ = 2;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
    }
};

// Specialise for double
template <> class LaunchTraits<double>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 4;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int nchan, int na, int ntime)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            nchan, na, ntime);
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
    int nsrc, int ntime, int na, int nchan)
{
    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;

    if(chan >= nchan || ant >= na || time >= ntime)
        { return; }

    // Simpler float and complex types
    typedef typename Traits::FT FT;
    typedef typename Traits::CT CT;

    typedef typename montblanc::kernel_policies<FT> Po;
    typedef typename montblanc::phase::LaunchTraits<FT> LTr;

    // Lightspeed
    constexpr FT lightspeed = 299792458;
    constexpr FT two_pi_over_c = FT(-2.0*M_PI/lightspeed);

    __shared__ typename Traits::uvw_type
        s_uvw[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];
    __shared__ typename Traits::frequency_type
        s_freq[LTr::BLOCKDIMX];

    // UVW coordinates vary by antenna and time, but not channel
    if(threadIdx.x == 0)
        { s_uvw[threadIdx.z][threadIdx.y] = uvw[time*na + ant]; }

    // Wavelengths vary by channel, not by time and antenna
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { s_freq[threadIdx.x] = frequency[chan]; }

    __syncthreads();

    // Iterate over sources
    for(int src=0; src < nsrc; ++src)
    {
        // Calculate the n coordinate
        typename Traits::lm_type r_lm = lm[src];
        FT n = Po::sqrt(FT(1.0) - r_lm.x*r_lm.x - r_lm.y*r_lm.y)
            - FT(1.0);

        // Calculate the real phase term
        FT real_phase = s_uvw[threadIdx.z][threadIdx.y].z*n
            + s_uvw[threadIdx.z][threadIdx.y].y*r_lm.y
            + s_uvw[threadIdx.z][threadIdx.y].x*r_lm.x;

        real_phase *= two_pi_over_c*s_freq[threadIdx.x];

        CT cplx_phase;
        Po::sincos(real_phase, &cplx_phase.y, &cplx_phase.x);

        int i = ((src*ntime + time)*na + ant)*nchan + chan;
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
        int ntime = in_uvw.dim_size(0);
        int na = in_uvw.dim_size(1);
        int nchan = in_frequency.dim_size(0);

        // Reason about our output shape
        tf::TensorShape complex_phase_shape({nsrc, ntime, na, nchan});

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
        dim3 blocks(LTr::block_size(nchan, na, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, nchan, na, ntime));

        //printf("Threads per block: X %d Y %d Z %d\n",
        //    blocks.x, blocks.y, blocks.z);

        //printf("Grid: X %d Y %d Z %d\n",
        //    grid.x, grid.y, grid.z);

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
            nsrc, ntime, na, nchan);
    }
};

} // namespace phase {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #define RIME_PHASE_OP_GPU_H