#if GOOGLE_CUDA

#ifndef RIME_EKB_SQRT_OP_GPU_CUH
#define RIME_EKB_SQRT_OP_GPU_CUH

#include "ekb_sqrt_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for floats and doubles
template <typename FT> struct LaunchTraits {};

template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

// CUDA kernel outline
template <typename Traits>
__global__ void rime_ekb_sqrt(
    const typename Traits::CT * complex_phase,
    const typename Traits::CT * bsqrt,
    const typename Traits::CT * ejones,
    typename Traits::CT * ant_jones,
    int nsrc, int ntime, int na, int nchan, int npol)
{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;
    //__shared__ FT buffer[LTr::BLOCKDIMX];

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan / npol;
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;
    int npolchan = nchan*npol;

    if(time > ntime || ant >= na || polchan > npolchan)
        { return; }

    int i;

    for(int src=0; src < nsrc; ++ src)
    {
        // Load in bsqrt
        int src_time = src*ntime + time;
        i = src_time*npolchan + polchan;
        CT brightness_sqrt = bsqrt[i];

        // Load in the complex phase
        int src_time_ant = src_time*na + ant;
        i = src_time_ant*nchan + chan;
        CT cplx_phase = complex_phase[i];

        // Load in the brightness square root and multiply the phase in
        montblanc::complex_multiply_in_place<FT>(cplx_phase, brightness_sqrt);

        i = src_time_ant*npolchan + polchan;
        CT E = ejones[i];

        // Load in the E Beam and multiply by KB
        montblanc::jones_multiply_4x4_in_place<FT>(E, cplx_phase);

        // Output it
        ant_jones[i] = E;
    }
}

// Specialise the EKBSqrt op for GPUs
template <typename FT, typename CT>
class EKBSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit EKBSqrt(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_complex_phase = context->input(0);
        const tf::Tensor & in_bsqrt = context->input(1);
        const tf::Tensor & in_ejones = context->input(2);

        // Extract problem dimensions
        int nsrc = in_complex_phase.dim_size(0);
        int ntime = in_complex_phase.dim_size(1);
        int na = in_complex_phase.dim_size(2);
        int nchan = in_complex_phase.dim_size(3);
        int npol = in_bsqrt.dim_size(3);
        int npolchan = nchan*npol;

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, npol});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr =  montblanc::kernel_traits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npolchan, na, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, npolchan, na, ntime));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Get pointers to flattened tensor data buffers
        auto complex_phase = reinterpret_cast<const typename Tr::CT *>(
            in_complex_phase.flat<CT>().data());
        auto bsqrt = reinterpret_cast<const typename Tr::CT *>(
            in_bsqrt.flat<CT>().data());
        auto ejones = reinterpret_cast<const typename Tr::CT *>(
            in_ejones.flat<CT>().data());
        auto ant_jones = reinterpret_cast<typename Tr::CT *>(
            ant_jones_ptr->flat<CT>().data());

        // Call the rime_ekb_sqrt CUDA kernel
        rime_ekb_sqrt<Tr><<<grid, block, 0, device.stream()>>>(
            complex_phase, bsqrt, ejones, ant_jones,
            nsrc, ntime, na, nchan, npol);
    }
};

MONTBLANC_EKB_SQRT_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_EKB_SQRT_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
