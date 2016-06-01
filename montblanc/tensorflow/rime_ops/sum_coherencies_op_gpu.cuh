#ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH
#define RIME_SUM_COHERENCIES_OP_GPU_CUH

#if GOOGLE_CUDA

#include "sum_coherencies_op.h"
#include <montblanc/abstraction.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace sumcoherencies {

// Traits class defined by float types
template <typename FT> class LaunchTraits;

// Specialise for float
template <> class LaunchTraits<float>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 32;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }        
};

// Specialise for double
template <> class LaunchTraits<double>
{
public:
    static constexpr int BLOCKDIMX = 32;
    static constexpr int BLOCKDIMY = 16;
    static constexpr int BLOCKDIMZ = 1;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }        
};

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

__constant__ const_data cdata;

template <typename Traits>
__global__ void rime_sum_coherencies(
    const typename Traits::uvw_type * uvw,
    const typename Traits::gauss_shape_type * gauss_shape,
    const typename Traits::sersic_shape_type  * sersic_shape,
    const typename Traits::frequency_type * frequency,
    const typename Traits::antenna_type * antenna1,
    const typename Traits::antenna_type * antenna2,
    const typename Traits::ant_jones_type * ant_jones,
    const typename Traits::flag_type * flag,
    const typename Traits::weight_type * weight,
    const typename Traits::gterm_type * gterm,
    const typename Traits::vis_type * observed_vis,
    const typename Traits::vis_type * in_model_vis,
    typename Traits::vis_type * out_model_vis)
{
    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int BL = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;

    if(TIME >= cdata.ntime || BL >= cdata.nbl || POLCHAN >= cdata.npolchan)
            { return; }

    // Helpful types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = montblanc::sumcoherencies::LaunchTraits<FT>;
    using Po = montblanc::kernel_policies<FT>;

    // Shared memory data structure
    __shared__ struct {
        typename Traits::uvw_type uvw[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];
        typename Traits::frequency_type freq[LTr::BLOCKDIMX];

        // Gaussian shape parameters
        FT el, em, eR;
        // Sersic shape parameters
        FT e1, e2, sersic_scale;
    } shared;

    // References
    FT & U = shared.uvw[threadIdx.z][threadIdx.y].x;
    FT & V = shared.uvw[threadIdx.z][threadIdx.y].y;
    FT & W = shared.uvw[threadIdx.z][threadIdx.y].z;

    int i;

    i = TIME*cdata.nbl + BL;
    int ANT1 = antenna1[i];
    int ANT2 = antenna2[i];

    // UVW coordinates vary by baseline and time, but not polarised channel
    if(threadIdx.x == 0)
    {
        // UVW, calculated from u_pq = u_p - u_q
        i = TIME*cdata.na + ANT2;
        shared.uvw[threadIdx.z][threadIdx.y] = uvw[i];

        i = TIME*cdata.na + ANT1;
        typename Traits::uvw_type ant1_uvw = uvw[i];
        U -= ant1_uvw.x;
        V -= ant1_uvw.y;
        W -= ant1_uvw.z;
    }

    // Wavelength varies by channel, but not baseline and time
    // TODO uses 4 times the actually required space, since
    // we don't need to store a frequency per polarisation
    if(threadIdx.y == 0 && threadIdx.z == 0)
        { shared.freq[threadIdx.x] = frequency[POLCHAN >> 2]; }

    // We process sources in batches, accumulating visibility values.
    // Visibilities are read in and written out at the start and end
    // of each batch respectively.

    // Initialise polarisation to zero if this is the first batch.
    // Otherwise, read in the visibilities from the previous batch.
    // CT polsum = {0.0, 0.0};
    // if(cdata.nsrc.lower_extent > 0)
    // {
    //     i = (TIME*cdata.nbl + BL)*cdata.npolchan + POLCHAN;
    //     polsum = model_vis[i];
    // }

    i = (TIME*cdata.nbl + BL)*cdata.npolchan + POLCHAN;
    CT polsum = in_model_vis[i];

    // Iterate over point sources
    int src_start = 0;
    int src_stop = cdata.npsrc.extent_size();

    for(int src=src_start; src < src_stop; ++src)
    {
        polsum.x += 1.0;
    }

    // Iterate over gaussian sources
    src_start = src_stop;
    src_stop += cdata.ngsrc.extent_size();

    for(int src=src_start; src < src_stop; ++src)
    {
        polsum.x += 1.0;
    }

    // Iterate over sersic sources
    src_start = src_stop;
    src_stop += cdata.nssrc.extent_size();

    for(int src=src_start; src < src_stop; ++src)
    {
        polsum.x += 1.0;
    }

    if(flag[i] > 0)
    {
        polsum.x = 0;
        polsum.y = 0;
    }

    i = (TIME*cdata.nbl + BL)*cdata.npolchan + POLCHAN;
    out_model_vis[i] = polsum;
}

template <typename FT, typename CT>
class RimeSumCoherencies<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    // Pointer to constant memory on the device
    const_data * d_cdata;
public:
    explicit RimeSumCoherencies(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        // Get device address of GPU constant data
        cudaError_t error = cudaGetSymbolAddress((void **)&d_cdata, cdata);

        if(error != cudaSuccess) {
            printf("Cuda Error: %s\n", cudaGetErrorString(error));
        }        
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        const tf::Tensor & in_uvw = context->input(0);
        const tf::Tensor & in_gauss_shape = context->input(1);
        const tf::Tensor & in_sersic_shape = context->input(2);
        const tf::Tensor & in_frequency = context->input(3);
        const tf::Tensor & in_antenna1 = context->input(4);
        const tf::Tensor & in_antenna2 = context->input(5);
        const tf::Tensor & in_ant_jones = context->input(6);
        const tf::Tensor & in_flag = context->input(7);
        const tf::Tensor & in_weight = context->input(8);
        const tf::Tensor & in_gterm = context->input(9);
        const tf::Tensor & in_obs_vis = context->input(10);
        const tf::Tensor & in_in_model_vis = context->input(11);

        OP_REQUIRES(context, in_uvw.dims() == 3 && in_uvw.dim_size(2) == 3,
            tf::errors::InvalidArgument(
                "uvw should be of shape (ntime, na, 3)"))

        OP_REQUIRES(context, in_obs_vis.dims() == 4 && in_obs_vis.dim_size(3) == 4,
            tf::errors::InvalidArgument(
                "obs_vis should be of shape (ntime, nbl, nchan, 4"))

        int ntime = in_obs_vis.dim_size(0);
        int nbl = in_obs_vis.dim_size(1);
        int nchan = in_obs_vis.dim_size(2);
        int npol = in_obs_vis.dim_size(3);
        int npolchan = nchan*npol;
        
        int ngsrc = in_gauss_shape.dim_size(0);
        int nssrc = in_sersic_shape.dim_size(0);
        int nsrc = in_ant_jones.dim_size(0);
        int na = in_ant_jones.dim_size(2);
        int npsrc = nsrc - ngsrc - nssrc;

        // Allocate an output tensor
        tf::TensorShape model_vis_shape({ntime, nbl, nchan, npol});
        tf::Tensor * model_vis_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, model_vis_shape, &model_vis_ptr));
        
        if(model_vis_ptr->NumElements() == 0)
            { return; }

        tf::TensorShape cdata_shape({sizeof(cdata)});
        tf::Tensor cdata_tensor;

        // TODO. Does this actually allocate pinned memory?
        tf::AllocatorAttributes pinned_allocator;
        pinned_allocator.set_on_host(true);
        pinned_allocator.set_gpu_compatible(true);

        // Allocate memory for the constant data
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT8, cdata_shape, &cdata_tensor,
            pinned_allocator));

        // Cast raw bytes to the constant data structure type
        const_data * cdata_ptr = reinterpret_cast<const_data *>(
            cdata_tensor.flat<uint8_t>().data());

        cdata_ptr->ntime = ntime;
        cdata_ptr->na = na;
        cdata_ptr->nbl = nbl;
        cdata_ptr->nchan = nchan;
        cdata_ptr->npolchan = npolchan;

        cdata_ptr->npsrc.local_size = npsrc;
        cdata_ptr->npsrc.global_size = npsrc;
        cdata_ptr->npsrc.lower_extent = 0;
        cdata_ptr->npsrc.upper_extent = npsrc;

        cdata_ptr->ngsrc.local_size = ngsrc;
        cdata_ptr->ngsrc.global_size = ngsrc;
        cdata_ptr->ngsrc.lower_extent = 0;
        cdata_ptr->ngsrc.upper_extent = ngsrc;

        cdata_ptr->nssrc.local_size = nssrc;
        cdata_ptr->nssrc.global_size = nssrc;
        cdata_ptr->nssrc.lower_extent = 0;
        cdata_ptr->nssrc.upper_extent = nssrc;

        cdata_ptr->nsrc.local_size = nsrc;
        cdata_ptr->nsrc.global_size = nsrc;
        cdata_ptr->nsrc.lower_extent = 0;
        cdata_ptr->nsrc.upper_extent = nsrc;

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        // Enqueue a copy of constant data to the device
        cudaMemcpyAsync(d_cdata, cdata_ptr, sizeof(cdata),
           cudaMemcpyHostToDevice, stream);

        using Tr = montblanc::kernel_traits<FT>;
        using LTr = LaunchTraits<FT>;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(npolchan, nbl, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, npolchan, nbl, ntime));

        auto uvw = reinterpret_cast<const typename Tr::uvw_type *>(
            in_uvw.flat<FT>().data());
        auto gauss_shape = reinterpret_cast<
            const typename Tr::gauss_shape_type *>(
                in_gauss_shape.flat<FT>().data());
        auto sersic_shape = reinterpret_cast<
            const typename Tr::sersic_shape_type *>(
                in_gauss_shape.flat<FT>().data());
        auto frequency = reinterpret_cast<
            const typename Tr::frequency_type *>(
                in_frequency.flat<FT>().data());
        auto antenna1 = reinterpret_cast<
            const typename Tr::antenna_type *>(
                in_antenna1.flat<int>().data());
        auto antenna2 = reinterpret_cast<
            const typename Tr::antenna_type *>(
                in_antenna2.flat<int>().data());
        auto antenna_jones = reinterpret_cast<
            const typename Tr::ant_jones_type *>(
                in_ant_jones.flat<CT>().data());
        auto flag = reinterpret_cast<
            const typename Tr::flag_type *>(
                in_flag.flat<uint8_t>().data());
        auto weight = reinterpret_cast<
            const typename Tr::weight_type *>(
                in_weight.flat<FT>().data());
        auto gterm = reinterpret_cast<
            const typename Tr::gterm_type *>(
                in_gterm.flat<CT>().data());
        auto observed_vis = reinterpret_cast<
            const typename Tr::vis_type *>(
                in_obs_vis.flat<CT>().data());
        auto in_model_vis = reinterpret_cast<
            const typename Tr::vis_type *>(
                in_in_model_vis.flat<CT>().data());
        auto out_model_vis = reinterpret_cast<
            typename Tr::vis_type *>(
                model_vis_ptr->flat<CT>().data());

        rime_sum_coherencies<Tr><<<grid, blocks, 0, stream>>>(
            uvw, gauss_shape, sersic_shape, frequency,
            antenna1, antenna2, antenna_jones, flag, weight,
            gterm, observed_vis, in_model_vis, out_model_vis);
    }
};

} // namespace sumcoherencies {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #define RIME_SUM_COHERENCIES_OP_GPU_CUH