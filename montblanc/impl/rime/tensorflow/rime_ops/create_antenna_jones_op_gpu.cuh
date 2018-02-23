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
    const int * arow_time_index,
    typename Traits::CT * ant_jones,
    int nsrc, int ntime, int narow, int nchan, int npol)
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan / npol;
    int pol = polchan & (npol-1);
    int arow = blockIdx.y*blockDim.y + threadIdx.y;
    int npolchan = nchan*npol;

    if(arow >= narow || polchan > npolchan)
        { return; }

    int i;

    __shared__ struct {
        CT fr[LTr::BLOCKDIMY][CREATE_ANTENNA_JONES_NPOL];
        int time_index[LTr::BLOCKDIMY];
    } shared;

    // Feed rotation varies by arow and polarisation
    // Polarisation is baked into the X dimension, so use the
    // first npol threads to load polarisation info
    if(threadIdx.x < npol)
    {
        i = arow*npol + pol;
        shared.fr[threadIdx.y][threadIdx.x] = feed_rotation[i];
    }

    // time_index varies by arow
    if(threadIdx.x == 0)
    {
        shared.time_index[threadIdx.y] = arow_time_index[arow];
    }

    __syncthreads();

    for(int src=0; src < nsrc; ++src)
    {
        // Load in bsqrt
        i = src*ntime + shared.time_index[threadIdx.y];
        CT brightness_sqrt = bsqrt[i*npolchan + polchan];

        // Load in the complex phase
        int i = (src*narow + arow)*nchan + chan;
        CT cplx_phase = complex_phase[i];

        // Multiply brightness square root into the complex phase
        montblanc::complex_multiply_in_place<FT>(cplx_phase, brightness_sqrt);

        // Load in the feed rotation and multiply by KB
        CT L = shared.fr[threadIdx.y][pol];

        montblanc::jones_multiply_4x4_in_place<FT>(L, cplx_phase);

        // Load in the E Beam and multiply by LKB
        i = (src*narow + arow)*npolchan + polchan;
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
        const tf::Tensor & in_arow_time_index = context->input(4);

        // Extract problem dimensions
        int nsrc = in_complex_phase.dim_size(0);
        int narow = in_complex_phase.dim_size(1);
        int ntime = in_bsqrt.dim_size(1);
        int nchan = in_complex_phase.dim_size(2);
        int npol = in_bsqrt.dim_size(3);
        int npolchan = nchan*npol;

        //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, npol == CREATE_ANTENNA_JONES_NPOL,
            tf::errors::InvalidArgument("Number of polarisations '",
                npol, "' does not equal '", CREATE_ANTENNA_JONES_NPOL, "'."));

        tf::TensorShape ant_jones_shape({nsrc, narow, nchan, npol});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr =  montblanc::kernel_traits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npolchan, narow, 1);
        dim3 grid(montblanc::grid_from_thread_block(block,
            npolchan, narow, 1));

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
        auto arow_time_index = reinterpret_cast<const int *>(
            in_arow_time_index.flat<int>().data());
        auto ant_jones = reinterpret_cast<typename Tr::CT *>(
            ant_jones_ptr->flat<CT>().data());

        // Call the rime_create_antenna_jones CUDA kernel
        rime_create_antenna_jones<Tr><<<grid, block, 0, device.stream()>>>(
            bsqrt, complex_phase, feed_rotation,
            ejones, arow_time_index, ant_jones,
            nsrc, ntime, narow, nchan, npol);
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
