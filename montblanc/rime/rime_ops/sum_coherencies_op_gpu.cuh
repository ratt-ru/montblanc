#if GOOGLE_CUDA

#ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH
#define RIME_SUM_COHERENCIES_OP_GPU_CUH

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "sum_coherencies_op.h"
#include "shapes.h"

#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

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
    static constexpr int BLOCKDIMX = 16;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;
};

template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 16;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;
};

// CUDA kernel outline
template <typename Traits>
__global__ void rime_sum_coherencies(
    const int * time_index,
    const typename Traits::antenna_type * antenna1,
    const typename Traits::antenna_type * antenna2,
    const typename Traits::ant_jones_type * ant_jones_1,
    const typename Traits::ant_jones_type * baseline_jones,
    const typename Traits::ant_jones_type * ant_jones_2,
    const typename Traits::vis_type * base_coherencies,
    typename Traits::vis_type * coherencies,
    int nsrc, int ntime, int nvrow, int na, int nchan, int npolchan)
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan >> 2;
    int vrow = blockIdx.y*blockDim.y + threadIdx.y;

    if(vrow >= nvrow || polchan >= npolchan)
        { return; }

    // Antenna indices for the baseline
    int ant1 = antenna1[vrow];
    int ant2 = antenna2[vrow];
    int time = time_index[vrow];

    int i;

    CT coherency = {0.0, 0.0};

    // Load in model visibilities
    if(base_coherencies != nullptr)
        { coherency = base_coherencies[vrow*npolchan + polchan]; }

    // Sum over visibilities
    for(int src=0; src < nsrc; ++src)
    {
        CT AJ1 = montblanc::jones_identity<FT>();;

        int base = src*ntime + time;

        // Load in antenna 1 jones
        if(ant_jones_1 != nullptr)
        {
            i = (base*na + ant1)*npolchan + polchan;
            AJ1 = ant_jones_1[i];
        }

        // Multiply in baseline jones
        if(baseline_jones != nullptr)
        {
            i = (src*nvrow + vrow)*npolchan + polchan;
            CT BLJ = baseline_jones[i];
            montblanc::jones_multiply_4x4_in_place<FT>(AJ1, BLJ);
        }

        // Multiply in antenna 2 jones
        if(ant_jones_2 != nullptr)
        {
            // Load antenna 2 jones
            i = (base*na + ant2)*npolchan + polchan;
            CT AJ2 = ant_jones_2[i];
            montblanc::jones_multiply_4x4_hermitian_transpose_in_place<FT>(AJ1, AJ2);
        }

        // Sum source coherency into model visibility
        coherency.x += AJ1.x;
        coherency.y += AJ1.y;
    }

    i = vrow*npolchan + polchan;
    // Write out the polarisation
    coherencies[i] = coherency;
}

// Specialise the SumCoherencies op for GPUs
template <typename FT, typename CT>
class SumCoherencies<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    TensorflowInputFacade<TFOpKernel> in_facade;

public:
    explicit SumCoherencies(tensorflow::OpKernelConstruction * ctx) :
        tensorflow::OpKernel(ctx),
        in_facade({"time_index", "antenna1", "antenna2",
                   "ant_jones_1",
                   "baseline_jones",
                   "ant_jones_2",
                   "base_coherencies"})
    {
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx));
    }

    void Compute(tensorflow::OpKernelContext * ctx) override
    {
        namespace tf = tensorflow;

        typename TensorflowInputFacade<TFOpKernel>::OpInputData op_data;
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx, &op_data));

        const tf::Tensor * time_index_ptr = nullptr;
        const tf::Tensor * antenna1_ptr = nullptr;
        const tf::Tensor * antenna2_ptr = nullptr;
        const tf::Tensor * ant_jones_1_ptr = nullptr;
        const tf::Tensor * baseline_jones_ptr = nullptr;
        const tf::Tensor * ant_jones_2_ptr = nullptr;
        const tf::Tensor * base_coherencies_ptr = nullptr;

        OP_REQUIRES_OK(ctx, op_data.get_tensor("time_index", 0,
                                                 &time_index_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna1", 0,
                                                 &antenna1_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna2", 0,
                                                 &antenna2_ptr));
        bool have_ant_1_jones = op_data.get_tensor("ant_jones_1", 0,
                                                 &ant_jones_1_ptr).ok();
        bool have_bl_jones = op_data.get_tensor("baseline_jones", 0,
                                                 &baseline_jones_ptr).ok();
        bool have_ant_2_jones = op_data.get_tensor("ant_jones_2", 0,
                                                 &ant_jones_2_ptr).ok();
        bool have_base = op_data.get_tensor("base_coherencies", 0,
                                                 &base_coherencies_ptr).ok();

        OP_REQUIRES(ctx, have_ant_1_jones || have_bl_jones || have_ant_2_jones,
            tf::errors::InvalidArgument("No Jones Terms were supplied"));

        int nvrow, nsrc, ntime = 0, na = 0, nchan, ncorr;
        OP_REQUIRES_OK(ctx, op_data.get_dim("row", &nvrow));
        OP_REQUIRES_OK(ctx, op_data.get_dim("source", &nsrc));
        // Without antenna jones terms, these may not be present
        op_data.get_dim("time", &ntime);
        op_data.get_dim("ant", &na);
        OP_REQUIRES_OK(ctx, op_data.get_dim("chan", &nchan));
        OP_REQUIRES_OK(ctx, op_data.get_dim("corr", &ncorr));

        int ncorrchan = nchan*ncorr;

        // Allocate an output tensor
        tf::Tensor * coherencies_ptr = nullptr;
        tf::TensorShape coherencies_shape = tf::TensorShape({
            nvrow, nchan, ncorr });
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, coherencies_shape, &coherencies_ptr));

        // Cast input into CUDA types defined within the Traits class
        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;


        auto time_index = reinterpret_cast<const int *>(
            time_index_ptr->flat<int>().data());
        auto antenna1 = reinterpret_cast<const typename Tr::antenna_type *>(
            antenna1_ptr->flat<int>().data());
        auto antenna2 = reinterpret_cast<const typename Tr::antenna_type *>(
            antenna2_ptr->flat<int>().data());
        auto ant_jones_1 = !have_ant_1_jones ? nullptr :
                    reinterpret_cast<const typename Tr::ant_jones_type *>(
                        ant_jones_1_ptr->flat<CT>().data());
        auto baseline_jones = !have_bl_jones ? nullptr :
                    reinterpret_cast<const typename Tr::ant_jones_type *>(
                        baseline_jones_ptr->flat<CT>().data());
        auto ant_jones_2 = !have_ant_2_jones ? nullptr :
                    reinterpret_cast<const typename Tr::ant_jones_type *>(
                        ant_jones_2_ptr->flat<CT>().data());
        auto base_coherencies = !have_base ? nullptr :
                    reinterpret_cast<const typename Tr::vis_type *>(
                        base_coherencies_ptr->flat<CT>().data());
        auto coherencies = reinterpret_cast<typename Tr::vis_type *>(
                        coherencies_ptr->flat<CT>().data());

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            ncorrchan, nvrow, 1);
        dim3 grid(montblanc::grid_from_thread_block(block,
            ncorrchan, nvrow, 1));

        // Get the GPU device
        const auto & device = ctx->eigen_device<GPUDevice>();

        // Call the rime_sum_coherencies CUDA kernel
        rime_sum_coherencies<Tr><<<grid, block, 0, device.stream()>>>(
            time_index, antenna1, antenna2,
            ant_jones_1,
            baseline_jones,
            ant_jones_2,
            base_coherencies, coherencies,
            nsrc, ntime, nvrow, na, nchan, ncorrchan);

        cudaError_t e = cudaPeekAtLastError();
        if(e != cudaSuccess) {
            OP_REQUIRES_OK(ctx,
                tf::errors::Internal("Cuda Failure ", __FILE__, __LINE__, " ",
                                             cudaGetErrorString(e)));
        }

    }
};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
