#ifndef RIME_E_BEAM_OP_GPU_CUH_
#define RIME_E_BEAM_OP_GPU_CUH_

#if GOOGLE_CUDA

#include "e_beam_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/brightness.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "rime_constant_structures.h"

namespace montblanc {
namespace ebeam {

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

typedef struct {
    dim_field nchan;
    dim_field npolchan;
} const_data;

} // namespace montblanc {
} // namespace ebeam {

namespace tensorflow {

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

// Number of polarisations handled by this kernel
constexpr int EBEAM_NPOL = 4;

// Constant GPU memory 
__constant__ montblanc::ebeam::const_data cdata;

// Get the current polarisation from the thread ID
template <typename Traits>
__device__ __forceinline__ int ebeam_pol()
    { return threadIdx.x & 0x3; }

template <typename Traits>
__global__ void rime_e_beam(
    const typename Traits::lm_type * lm,
    const typename Traits::point_error_type * point_errors,
    const typename Traits::antenna_scale_type * antenna_scaling,
    const typename Traits::CT * e_beam,
    typename Traits::CT * jones,
    typename Traits::FT parallactic_angle,
    typename Traits::FT beam_ll, typename Traits::FT beam_lm,
    typename Traits::FT beam_ul, typename Traits::FT beam_um,
    int nsrc, int ntime, int na, int npolchan)
{
    // Simpler float and complex types
    typedef typename Traits::FT FT;
    typedef typename Traits::CT CT;

    typedef typename montblanc::kernel_policies<FT> Po;
    typedef typename montblanc::ebeam::LaunchTraits<FT> LTr;

    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;
    constexpr int BLOCKCHANS = LTr::BLOCKDIMX >> 2;

    if(POLCHAN == 0 && ANT == 0 && SRC == 0)
    {
        printf("nchan.global_size=%d\n", cdata.nchan.global_size);
    }

    if(SRC >= nsrc || ANT >= na || POLCHAN >= npolchan)
        return;

    __shared__ typename Traits::lm_type
        s_lm0[LTr::BLOCKDIMZ];
    __shared__ typename Traits::point_error_type
        s_lmd[LTr::BLOCKDIMY][BLOCKCHANS];
    __shared__ typename Traits::antenna_scale_type
        s_ab[LTr::BLOCKDIMY][BLOCKCHANS];

}

template <typename FT, typename CT>
class RimeEBeam<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    // Pointer to constant memory on the device
    montblanc::rime_const_data * d_cdata;

public:
    explicit RimeEBeam(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        printf("Getting constant data address\n");
        // Get device address of GPU constant data
        cudaError_t error = cudaGetSymbolAddress((void **)&d_cdata, cdata);

        if(error != cudaSuccess) {
            printf("Cuda Error: %s\n", cudaGetErrorString(error));
        }

        printf("Got constant data address%\n");
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_point_errors = context->input(1);
        const tf::Tensor & in_antenna_scaling = context->input(2);
        const tf::Tensor & in_E_beam = context->input(3);
        const tf::Tensor & in_parallactic_angle = context->input(4);
        const tf::Tensor & in_beam_ll = context->input(5);
        const tf::Tensor & in_beam_lm = context->input(6);
        const tf::Tensor & in_beam_ul = context->input(7);
        const tf::Tensor & in_beam_um = context->input(8);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument("lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_point_errors.dims() == 4
            && in_point_errors.dim_size(3) == 2,
            tf::errors::InvalidArgument("point_errors should be of shape "
                                        "(ntime, na, nchan, 2)"))

        OP_REQUIRES(context, in_antenna_scaling.dims() == 3
            && in_antenna_scaling.dim_size(2) == 2,
            tf::errors::InvalidArgument("antenna_scaling should be of shape "
                                        "(na, nchan, 2)"))

        OP_REQUIRES(context, in_E_beam.dims() == 4
            && in_E_beam.dim_size(3) == 4,
            tf::errors::InvalidArgument("E_Beam should be of shape "
                                        "(beam_lw, beam_mh, beam_nud, 4)"))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_parallactic_angle.shape()),
            tf::errors::InvalidArgument("parallactic_angle is not scalar: ",
                in_parallactic_angle.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ll.shape()),
            tf::errors::InvalidArgument("in_beam_ll is not scalar: ",
                in_beam_ll.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_lm.shape()),
            tf::errors::InvalidArgument("in_beam_lm is not scalar: ",
                in_beam_lm.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_ul.shape()),
            tf::errors::InvalidArgument("in_beam_ul is not scalar: ",
                in_beam_ul.shape().DebugString()))

        OP_REQUIRES(context, tf::TensorShapeUtils::IsScalar(
                in_beam_um.shape()),
            tf::errors::InvalidArgument("in_beam_um is not scalar: ",
                in_beam_um.shape().DebugString()))

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int ntime = in_point_errors.dim_size(0);
        int na = in_point_errors.dim_size(1);
        int nchan = in_point_errors.dim_size(2);
        int npolchan = nchan*EBEAM_NPOL;

        // Reason about our output shape
        // Create a pointer for the jones result
        tf::TensorShape jones_shape({nsrc, ntime, na, nchan, 4});
        tf::Tensor * jones_ptr = nullptr;

        // Allocate memory for the jones
        OP_REQUIRES_OK(context, context->allocate_output(
            0, jones_shape, &jones_ptr));

        if (jones_ptr->NumElements() == 0)
            { return; }

        tf::TensorShape cdata_shape({sizeof(cdata)});
        tf::Tensor cdata_tensor;

        printf("Allocating constant data\n");

        // TODO. Does this actually allocate pinned memory?
        tensorflow::AllocatorAttributes pinned_allocator;
        pinned_allocator.set_on_host(true);
        pinned_allocator.set_gpu_compatible(true);

        // Allocate memory for the constant data
        OP_REQUIRES_OK(context, context->allocate_temp(
            DT_UINT8, cdata_shape, &cdata_tensor,
            pinned_allocator));

        printf("Allocated constant data\n");

        // Cast raw bytes to the constant data structure type
        montblanc::ebeam::const_data * cdata_ptr = 
            reinterpret_cast<montblanc::ebeam::const_data *>(
                cdata_tensor.flat<uint8_t>().data());

        printf("Accessing constant data\n");

        cdata_ptr->nchan.global_size = nchan;
        cdata_ptr->nchan.local_size = nchan;
        cdata_ptr->nchan.lower_extent = 0;
        cdata_ptr->nchan.upper_extent = nchan;

        cdata_ptr->npolchan.global_size = npolchan;
        cdata_ptr->npolchan.local_size = npolchan;
        cdata_ptr->npolchan.lower_extent = 0;
        cdata_ptr->npolchan.upper_extent = npolchan;


        printf("Accessed constant data\n");

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        // Enqueue a copy of constant data to the device
        cudaMemcpyAsync(d_cdata, cdata_ptr, sizeof(cdata),
           cudaMemcpyHostToDevice, stream);

        typedef montblanc::kernel_traits<FT> Tr;
        typedef typename montblanc::ebeam::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(npolchan, na, nsrc));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, npolchan, na, nsrc));

        // Cast to the cuda types expected by the kernel
        auto lm = reinterpret_cast<const typename Tr::lm_type *>(
            in_lm.flat<FT>().data());
        auto point_errors = reinterpret_cast<
            const typename Tr::point_error_type *>(
                in_point_errors.flat<FT>().data());
        auto antenna_scaling = reinterpret_cast<
            const typename Tr::antenna_scale_type *>(
                in_antenna_scaling.flat<FT>().data());
        auto E_beam = reinterpret_cast<const typename Tr::CT *>(
            in_E_beam.flat<CT>().data());
        auto jones = reinterpret_cast<typename Tr::CT *>(
            jones_ptr->flat<CT>().data());

        printf("Getting constants\n");
        FT parallactic_angle = in_parallactic_angle.tensor<FT, 0>()(0);
        FT beam_ll = in_beam_ll.tensor<FT, 0>()(0);
        FT beam_lm = in_beam_lm.tensor<FT, 0>()(0);
        FT beam_ul = in_beam_ul.tensor<FT, 0>()(0);
        FT beam_um = in_beam_um.tensor<FT, 0>()(0);

        printf("Calling kernel\n");

        rime_e_beam<Tr><<<grid, blocks, 0, stream>>>(
            lm, point_errors, antenna_scaling, E_beam,
            jones, parallactic_angle,
            beam_ll, beam_lm, beam_ul, beam_um,
            nsrc, ntime, na, npolchan);

    }
};

} // namespace tensorflow {

#endif // #if GOOGLE_CUDA

#endif // #ifndef RIME_E_BEAM_OP_GPU_CUH_