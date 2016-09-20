#if GOOGLE_CUDA

#ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH
#define RIME_SUM_COHERENCIES_OP_GPU_CUH

#include "sum_coherencies_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

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
__global__ void rime_sum_coherencies(
    const typename Traits::antenna_type * antenna1,
    const typename Traits::antenna_type * antenna2,
    const typename Traits::FT * shape,
    const typename Traits::ant_jones_type * ant_jones,
    const typename Traits::neg_ant_jones_type * neg_ant_jones,
    const typename Traits::flag_type * flag,
    const typename Traits::gterm_type * gterm,
    const typename Traits::vis_type * model_vis_in,
    bool apply_dies,
    typename Traits::vis_type * model_vis_out,
    int nsrc, int ntime, int nbl, int na, int nchan, int npolchan)
{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;
    //__shared__ FT buffer[LTr::BLOCKDIMX];

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan >> 2;
    int bl = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;

    if(time >= ntime || bl >= nbl || polchan >= npolchan)
        { return; }

    // Antenna indices for the baseline
    int i = time*nbl + bl;
    int ant1 = antenna1[i];
    int ant2 = antenna2[i];

    // Load in model visibilities
    i = (time*nbl + bl)*npolchan + polchan;
    CT model_vis = model_vis_in[i];

    // Sum over visibilities
    for(int src=0; src < nsrc; ++src)
    {
        int base = src*ntime + time;

        // Load in shape value
        i = (base*nbl + bl)*nchan + chan;
        FT shape_ = shape[i];
        // Load in antenna 1 jones
        i = (base*na + ant1)*npolchan + polchan;
        CT J1 = ant_jones[i];
        // Load antenna 2 jones
        i = (base*na + ant2)*npolchan + polchan;
        CT J2 = ant_jones[i];

        // Multiply shape factor into antenna 2 jones
        J2.x *= shape_; J2.y *= shape_;

        // Multiply jones matrices, result into J1
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<FT>(
            J1, J2);

        // Load in and apply in sign inversions stemming from
        // cholesky decompositions that must be applied.
        FT sign = FT(neg_ant_jones[base]);
        J1.x *= sign;
        J1.y *= sign;

        // Sum source coherency into model visibility
        model_vis.x += J1.x;
        model_vis.y += J1.y;
    }

    // Apply Direction Independent Effects if required
    if(apply_dies)
    {
        // Multiply the visibility by antenna 1's g term
        i = (time*na + ant1)*npolchan + polchan;
        CT ant1_gterm = gterm[i];
        montblanc::jones_multiply_4x4_in_place<FT>(
            ant1_gterm, model_vis);

        // Shift
        model_vis.x = ant1_gterm.x;
        model_vis.y = ant1_gterm.y;

        // Multiply the visibility by antenna 2's g term
        i = (time*na + ant2)*npolchan + polchan;
        CT ant2_gterm = gterm[i];
        montblanc::jones_multiply_4x4_hermitian_transpose_in_place<FT>(
            model_vis, ant2_gterm);
    }

    i = (time*nbl + bl)*npolchan + polchan;

    // If flags apply, zero out the polarisation
    if(flag[i])
    {
        model_vis.x = 0.0;
        model_vis.y = 0.0;
    }

    // Write out the polarisation
    model_vis_out[i] = model_vis;
}

// Specialise the SumCoherencies op for GPUs
template <typename FT, typename CT>
class SumCoherencies<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit SumCoherencies(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_antenna1 = context->input(0);
        const tf::Tensor & in_antenna2 = context->input(1);
        const tf::Tensor & in_shape = context->input(2);
        const tf::Tensor & in_ant_jones = context->input(3);
        const tf::Tensor & in_neg_ant_jones = context->input(4);
        const tf::Tensor & in_flag = context->input(5);
        const tf::Tensor & in_gterm = context->input(6);
        const tf::Tensor & in_model_vis_in = context->input(7);
        const tf::Tensor & in_apply_dies = context->input(8);

        int nsrc = in_shape.dim_size(0);
        int ntime = in_shape.dim_size(1);
        int nbl = in_shape.dim_size(2);
        int nchan = in_shape.dim_size(3);
        int na = in_ant_jones.dim_size(2);
        int npol = in_ant_jones.dim_size(4);
        int npolchan = nchan*npol;

        // Allocate an output tensor
        tf::Tensor * model_vis_out_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, in_model_vis_in.shape(), &model_vis_out_ptr));

        // Cast input into CUDA types defined within the Traits class
        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;

        auto antenna1 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna1.flat<int>().data());
        auto antenna2 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna2.flat<int>().data());
        auto shape = reinterpret_cast<const typename Tr::FT *>(
            in_shape.flat<FT>().data());
        auto ant_jones = reinterpret_cast<const typename Tr::ant_jones_type *>(
            in_ant_jones.flat<CT>().data());
        auto neg_ant_jones = reinterpret_cast<const typename Tr::neg_ant_jones_type *>(
            in_neg_ant_jones.flat<tf::int8>().data());
        auto flag = reinterpret_cast<const typename Tr::flag_type *>(
            in_flag.flat<tf::uint8>().data());
        auto gterm = reinterpret_cast<const typename Tr::gterm_type *>(
            in_gterm.flat<CT>().data());
        auto model_vis_in = reinterpret_cast<const typename Tr::vis_type *>(
            in_model_vis_in.flat<CT>().data());
        auto model_vis_out = reinterpret_cast<typename Tr::vis_type *>(
            model_vis_out_ptr->flat<CT>().data());
        auto apply_dies = in_apply_dies.tensor<bool, 0>()(0);

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npolchan, nbl, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, npolchan, nbl, ntime));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_sum_coherencies CUDA kernel
        rime_sum_coherencies<Tr><<<grid, block, 0, device.stream()>>>(
            antenna1, antenna2, shape, ant_jones, neg_ant_jones,
            flag, gterm, model_vis_in, apply_dies, model_vis_out,
            nsrc, ntime, nbl, na, nchan, npolchan);
    }
};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
