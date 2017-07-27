#if GOOGLE_CUDA

#ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH
#define RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#include "create_antenna_jones_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN

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
__global__ void rime_create_antenna_jones(
    const typename Traits::CT * bsqrt,
    const typename Traits::CT * complex_phase,
    const typename Traits::CT * feed_rotation,
    const typename Traits::CT * ejones,
    typename Traits::CT * ant_jones,
    int nsrc, int ntime, int na, int nchan, int npol)
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan / npol;
    int pol = polchan & (npol-1);
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;
    int npolchan = nchan*npol;

    if(time > ntime || ant >= na || polchan > npolchan)
        { return; }

    int i;

    __shared__ struct {
        CT fr[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][CREATE_ANTENNA_JONES_NPOL];
    } shared;

    // Feed rotation varies by time, antenna and polarisation
    // Polarisation is baked into the X dimension, so use the
    // first npol threads to load polarisation info
    if(threadIdx.x < npol)
    {
        i = (time*na + ant)*npol + pol;
        shared.fr[threadIdx.z][threadIdx.y][threadIdx.x] = feed_rotation[i];
    }

    __syncthreads();

    for(int src=0; src < nsrc; ++src)
    {
        // Load in bsqrt
        int src_time = src*ntime + time;
        i = src_time*npolchan + polchan;
        CT brightness_sqrt = bsqrt[i];

        // Load in the complex phase
        int src_time_ant = src_time*na + ant;
        i = src_time_ant*nchan + chan;
        CT cplx_phase = complex_phase[i];

        // Multiply brightness square root into the complex phase
        montblanc::complex_multiply_in_place<FT>(cplx_phase, brightness_sqrt);

        // Load in the feed rotation and multiply by KB
        CT L = shared.fr[threadIdx.z][threadIdx.y][pol];

        montblanc::jones_multiply_4x4_in_place<FT>(L, cplx_phase);

        // Load in the E Beam and multiply by LKB
        i = src_time_ant*npolchan + polchan;
        CT E = ejones[i];

        montblanc::jones_multiply_4x4_in_place<FT>(E, L);

        // Output final per antenna value
        ant_jones[i] = E;
    }
}

// Specialise the CreateAntennaJones op for GPUs
template <typename FT, typename CT>
class CreateAntennaJones<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_bsqrt = context->input(0);
        const tf::Tensor & in_complex_phase = context->input(1);
        const tf::Tensor & in_feed_rotation = context->input(2);
        const tf::Tensor & in_ejones = context->input(3);

        // Extract problem dimensions
        int nsrc = in_complex_phase.dim_size(0);
        int ntime = in_complex_phase.dim_size(1);
        int na = in_complex_phase.dim_size(2);
        int nchan = in_complex_phase.dim_size(3);
        int npol = in_bsqrt.dim_size(3);
        int npolchan = nchan*npol;

        //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, npol == CREATE_ANTENNA_JONES_NPOL,
            tf::errors::InvalidArgument("Number of polarisations '",
                npol, "' does not equal '", CREATE_ANTENNA_JONES_NPOL, "'."));

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
        auto bsqrt = reinterpret_cast<const typename Tr::CT *>(
            in_bsqrt.flat<CT>().data());
        auto complex_phase = reinterpret_cast<const typename Tr::CT *>(
            in_complex_phase.flat<CT>().data());
        auto feed_rotation = reinterpret_cast<const typename Tr::CT *>(
            in_feed_rotation.flat<CT>().data());
        auto ejones = reinterpret_cast<const typename Tr::CT *>(
            in_ejones.flat<CT>().data());
        auto ant_jones = reinterpret_cast<typename Tr::CT *>(
            ant_jones_ptr->flat<CT>().data());

        // Call the rime_create_antenna_jones CUDA kernel
        rime_create_antenna_jones<Tr><<<grid, block, 0, device.stream()>>>(
            bsqrt, complex_phase, feed_rotation, ejones, ant_jones,
            nsrc, ntime, na, nchan, npol);
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
