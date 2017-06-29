#if GOOGLE_CUDA

#ifndef RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH
#define RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH

#include "post_process_visibilities_op.h"
#include <cub/cub/cub.cuh>
#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN;
MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
template <typename FT> struct LaunchTraits {};

// Specialise for float
template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};

// Specialise for double
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;
};


// CUDA kernel outline
template <typename Traits>
__global__ void rime_post_process_visibilities(
    const typename Traits::antenna_type * in_antenna1,
    const typename Traits::antenna_type * in_antenna2,
    const typename Traits::die_type * in_die,
    const typename Traits::flag_type * in_flag,
    const typename Traits::weight_type * in_weight,
    const typename Traits::vis_type * in_base_vis,
    const typename Traits::vis_type * in_model_vis,
    const typename Traits::vis_type * in_observed_vis,
    typename Traits::vis_type * out_final_vis,
    typename Traits::FT * out_chi_squared_terms,
    int ntime, int nbl, int na, int npolchan)

{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;

    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT>;

    int time = blockIdx.z*blockDim.z + threadIdx.z;
    int bl = blockIdx.y*blockDim.y + threadIdx.y;
    int polchan = blockIdx.x*blockDim.x + threadIdx.x;

    // Guard problem extents
    if(time >= ntime || bl >= nbl || polchan >= npolchan)
        { return; }

    // Antenna indices for the baseline
    int i = time*nbl + bl;
    int ant1 = in_antenna1[i];
    int ant2 = in_antenna2[i];

    // Load in model, observed visibilities, flags and weights
    i = (time*nbl + bl)*npolchan + polchan;
    CT base_vis = in_base_vis[i];
    CT model_vis = in_model_vis[i];
    CT diff_vis = in_observed_vis[i];
    FT weight = in_weight[i];
    // Flag multiplier used to zero flagged visibility points
    FT flag_mul = FT(in_flag[i] == 0);

    // Multiply the visibility by antenna 1's g term
    i = (time*na + ant1)*npolchan + polchan;
    CT ant1_die = in_die[i];
    montblanc::jones_multiply_4x4_in_place<FT>(
        ant1_die, model_vis);

    // Shift result
    model_vis.x = ant1_die.x;
    model_vis.y = ant1_die.y;

    // Multiply the visibility by antenna 2's g term
    i = (time*na + ant2)*npolchan + polchan;
    CT ant2_die = in_die[i];
    montblanc::jones_multiply_4x4_hermitian_transpose_in_place<FT>(
        model_vis, ant2_die);

    // Add any base visibilities
    model_vis.x += base_vis.x;
    model_vis.y += base_vis.y;

    // Subtract model visibilities from observed visibilities
    diff_vis.x -= model_vis.x;
    diff_vis.y -= model_vis.y;

    FT chi_squared_term = diff_vis.x*diff_vis.x + diff_vis.y*diff_vis.y;
    chi_squared_term *= weight*flag_mul;

    // Zero flagged visibilities
    model_vis.x *= flag_mul;
    model_vis.y *= flag_mul;

    i = (time*nbl + bl)*npolchan + polchan;
    out_final_vis[i] = model_vis;
    out_chi_squared_terms[i] = chi_squared_term;
}

// Specialise the PostProcessVisibilities op for GPUs
template <typename FT, typename CT>
class PostProcessVisibilities<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit PostProcessVisibilities(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_antenna1 = context->input(0);
        const auto & in_antenna2 = context->input(1);
        const auto & in_die = context->input(2);
        const auto & in_flag = context->input(3);
        const auto & in_weight = context->input(4);
        const auto & in_base_vis = context->input(5);
        const auto & in_model_vis = context->input(6);
        const auto & in_observed_vis = context->input(7);

        int ntime = in_model_vis.dim_size(0);
        int nbl = in_model_vis.dim_size(1);
        int nchan = in_model_vis.dim_size(2);
        int npol = in_model_vis.dim_size(3);
        int npolchan = npol*nchan;
        int na = in_die.dim_size(1);

        using LTr = LaunchTraits<FT>;

        // Allocate output tensors
        // Allocate space for output tensor 'final_vis'
        tf::Tensor * final_vis_ptr = nullptr;
        tf::TensorShape final_vis_shape = tf::TensorShape({
            ntime, nbl, nchan, npol });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, final_vis_shape, &final_vis_ptr));

       // Allocate space for output tensor 'chi_squared'
        tf::Tensor * chi_squared_ptr = nullptr;
        tf::TensorShape chi_squared_shape = tf::TensorShape({ });
        OP_REQUIRES_OK(context, context->allocate_output(
            1, chi_squared_shape, &chi_squared_ptr));

        // Get pointers to flattened tensor data buffers
        typedef montblanc::kernel_traits<FT> Tr;

        auto fin_antenna1 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna1.flat<tensorflow::int32>().data());
        auto fin_antenna2 = reinterpret_cast<const typename Tr::antenna_type *>(
            in_antenna2.flat<tensorflow::int32>().data());
        auto fin_die = reinterpret_cast<const typename Tr::die_type *>(
            in_die.flat<CT>().data());
        auto fin_flag = reinterpret_cast<const typename Tr::flag_type *>(
            in_flag.flat<tensorflow::uint8>().data());
        auto fin_weight = reinterpret_cast<const typename Tr::weight_type *>(
            in_weight.flat<FT>().data());
        auto fin_base_vis = reinterpret_cast<const typename Tr::vis_type *>(
            in_base_vis.flat<CT>().data());
        auto fin_model_vis = reinterpret_cast<const typename Tr::vis_type *>(
            in_model_vis.flat<CT>().data());
        auto fin_observed_vis = reinterpret_cast<const typename Tr::vis_type *>(
            in_observed_vis.flat<CT>().data());
        auto fout_final_vis = reinterpret_cast<typename Tr::vis_type *>(
            final_vis_ptr->flat<CT>().data());
        auto fout_chi_squared = chi_squared_ptr->flat<FT>().data();

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Create a GPU Allocator
        tf::AllocatorAttributes gpu_allocator;
        gpu_allocator.set_gpu_compatible(true);

        // Make a tensor to hold 'chi_squared' terms
        // These will be reduced into chi_squared
        tf::Tensor chi_squared_terms;
        tf::TensorShape chi_squared_terms_shape = tf::TensorShape({
            ntime, nbl, nchan, npol });
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DataTypeToEnum<FT>::value, chi_squared_terms_shape,
            &chi_squared_terms, gpu_allocator));

        FT * fout_chi_squared_terms = chi_squared_terms.flat<FT>().data();

        // Find out the size of the temporary storage buffer
        // cub needs to perform the reduction. nullptr invokes
        // this use case
        std::size_t temp_storage_bytes = 0;

        cub::DeviceReduce::Sum(nullptr, temp_storage_bytes,
            fout_chi_squared_terms, fout_chi_squared,
            chi_squared_terms.NumElements(), device.stream());

        // Make a tensor to hold temporary cub::DeviceReduce::Sum storage
        tf::Tensor temp_storage;
        tf::TensorShape temp_storage_shape = tf::TensorShape({
            (long long)temp_storage_bytes });
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT8, temp_storage_shape,
            &temp_storage, gpu_allocator));


        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npolchan, nbl, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, npolchan, nbl, ntime));

        // Call the rime_post_process_visibilities CUDA kernel
        rime_post_process_visibilities<Tr>
            <<<grid, block, 0, device.stream()>>>(
                fin_antenna1,
                fin_antenna2,
                fin_die,
                fin_flag,
                fin_weight,
                fin_base_vis,
                fin_model_vis,
                fin_observed_vis,
                fout_final_vis,
                fout_chi_squared_terms,
                ntime, nbl, na, npolchan);

        // Perform a reduction on the chi squared terms
        tf::uint8 * temp_storage_ptr = temp_storage.flat<tf::uint8>().data();
        cub::DeviceReduce::Sum(temp_storage_ptr, temp_storage_bytes,
            fout_chi_squared_terms, fout_chi_squared,
            chi_squared_terms.NumElements(), device.stream());
    }
};

MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA