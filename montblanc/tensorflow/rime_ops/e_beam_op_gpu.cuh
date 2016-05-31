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


// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;    

// Constant GPU memory 
__constant__ montblanc::ebeam::const_data cdata;

// Get the current polarisation from the thread ID
template <typename Traits>
__device__ __forceinline__ int ebeam_pol()
    { return threadIdx.x & 0x3; }

template <typename Traits, typename Policies>
__device__ __forceinline__
void trilinear_interpolate(
    typename Traits::CT & sum,
    typename Traits::FT & abs_sum,
    const typename Traits::CT * E_beam,
    float gl, float gm, float gchan,
    const typename Traits::FT & weight)
{
    // If this source is outside the cube, do nothing
    if(gl < 0 || gl >= cdata.beam_lw || gm < 0 || gm >= cdata.beam_mh)
        { return; }

    int i = ((int(gl)*cdata.beam_mh + int(gm))*cdata.beam_nud +
        int(gchan))*EBEAM_NPOL + ebeam_pol<Traits>();

    // Perhaps unnecessary as long as BLOCKDIMX is 32
    typename Traits::CT data = cub::ThreadLoad<cub::LOAD_LDG>(E_beam + i);
    sum.x += weight*data.x;
    sum.y += weight*data.y;
    abs_sum += weight*Policies::abs(data);
}

template <typename Traits>
__global__ void rime_e_beam(
    const typename Traits::lm_type * lm,
    const typename Traits::point_error_type * point_errors,
    const typename Traits::antenna_scale_type * antenna_scaling,
    const typename Traits::CT * e_beam,
    typename Traits::CT * jones,
    typename Traits::FT parallactic_angle,
    typename Traits::FT beam_ll, typename Traits::FT beam_lm,
    typename Traits::FT beam_ul, typename Traits::FT beam_um)
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

    if(SRC >= cdata.nsrc || ANT >= cdata.na || POLCHAN >= cdata.npolchan.extent_size())
        return;

    __shared__ typename Traits::lm_type
        s_lm0[LTr::BLOCKDIMZ];
    __shared__ typename Traits::point_error_type
        s_lmd[LTr::BLOCKDIMY][BLOCKCHANS];
    __shared__ typename Traits::antenna_scale_type
        s_ab[LTr::BLOCKDIMY][BLOCKCHANS];

    int i;

    // LM coordinates vary by source only,
    // not antenna or polarised channel
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        i = SRC;   s_lm0[threadIdx.z] = lm[i];
    }

    // Antenna scaling factors vary by antenna and channel,
    // but not source or timestep
    if(threadIdx.z == 0 && ebeam_pol<Traits>() == 0)
    {
        int blockchan = threadIdx.x >> 2;
        i = ANT*cdata.nchan.extent_size() + (POLCHAN >> 2);
        s_ab[threadIdx.y][blockchan] = antenna_scaling[i];
    }

    __syncthreads();

    for(int TIME=0; TIME < cdata.ntime; ++TIME)
    {
        // Pointing errors vary by time, antenna and channel,
        // but not source
        if(threadIdx.z == 0 && (threadIdx.x & 0x3) == 0)
        {
            int blockchan = threadIdx.x >> 2;
            i = (TIME*cdata.na + ANT)*cdata.nchan.extent_size() + (POLCHAN >> 2);
            s_lmd[threadIdx.y][blockchan] = point_errors[i];
        }

        __syncthreads();

        // Figure out how far the source has
        // rotated within the beam
        FT sint, cost;
        Po::sincos(parallactic_angle*TIME, &sint, &cost);

        // Rotate the source
        FT l = s_lm0[threadIdx.z].x*cost - s_lm0[threadIdx.z].y*sint;
        FT m = s_lm0[threadIdx.z].x*sint + s_lm0[threadIdx.z].y*cost;

        // Add the pointing errors for this antenna.
        int blockchan = threadIdx.x >> 2;
        l += s_lmd[threadIdx.y][blockchan].x;
        m += s_lmd[threadIdx.y][blockchan].y;

        // Multiply by the antenna scaling factors.
        l *= s_ab[threadIdx.y][blockchan].x;
        m *= s_ab[threadIdx.y][blockchan].y;

        // Compute grid position and difference from
        // actual position for the source at each channel
        l = FT(cdata.beam_lw-1) * (l - beam_ll) / (beam_ul - beam_ll);
        float gl = floorf(l);
        float ld = l - gl;

        m = FT(cdata.beam_mh-1) * (m - beam_lm) / (beam_um - beam_lm);
        float gm = floorf(m);
        float md = m - gm;

        // Work out where we are in the beam cube.
        // POLCHAN >> 2 is our position in the local channel space
        // Add this to the lower extent in the global channel space
        float chan = float(cdata.beam_nud-1) * float(POLCHAN>>2 + cdata.nchan.lower_extent)
            / float(cdata.nchan.global_size);
        float gchan = floorf(chan);
        float chd = chan - gchan;

        CT sum = Po::make_ct(0.0, 0.0);
        FT abs_sum = FT(0.0);

        // A simplified bilinear weighting is used here. Given
        // point x between points x1 and x2, with function f
        // provided values f(x1) and f(x2) at these points.
        //
        // x1 ------- x ---------- x2
        //
        // Then, the value of f can be approximated using the following:
        // f(x) ~= f(x1)(x2-x)/(x2-x1) + f(x2)(x-x1)/(x2-x1)
        //
        // Note how the value f(x1) is weighted with the distance
        // from the opposite point (x2-x).
        //
        // As we are interpolating on a grid, we have the following
        // 1. (x2 - x1) == 1
        // 2. (x - x1)  == 1 - 1 + (x - x1)
        //              == 1 - (x2 - x1) + (x - x1)
        //              == 1 - (x2 - x)
        // 2. (x2 - x)  == 1 - 1 + (x2 - x)
        //              == 1 - (x2 - x1) + (x2 - x)
        //              == 1 - (x - x1)
        //
        // Extending the above to 3D, we have
        // f(x,y,z) ~= f(x1,y1,z1)(x2-x)(y2-y)(z2-z) + ...
        //           + f(x2,y2,z2)(x-x1)(y-y1)(z-z1)
        //
        // f(x,y,z) ~= f(x1,y1,z1)(1-(x-x1))(1-(y-y1))(1-(z-z1)) + ...
        //           + f(x2,y2,z2)   (x-x1)    (y-y1)    (z-z1)

        // Load in the complex values from the E beam
        // at the supplied coordinate offsets.
        // Save the sum of abs in sum.real
        // and the sum of args in sum.imag
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 0.0f, gm + 0.0f, gchan + 0.0f,
            (1.0f-ld)*(1.0f-md)*(1.0f-chd));
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 1.0f, gm + 0.0f, gchan + 0.0f,
            ld*(1.0f-md)*(1.0f-chd));
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 0.0f, gm + 1.0f, gchan + 0.0f,
            (1.0f-ld)*md*(1.0f-chd));
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 1.0f, gm + 1.0f, gchan + 0.0f,
            ld*md*(1.0f-chd));

        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 0.0f, gm + 0.0f, gchan + 1.0f,
            (1.0f-ld)*(1.0f-md)*chd);
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 1.0f, gm + 0.0f, gchan + 1.0f,
            ld*(1.0f-md)*chd);
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 0.0f, gm + 1.0f, gchan + 1.0f,
            (1.0f-ld)*md*chd);
        trilinear_interpolate<Traits, Po>(sum, abs_sum, e_beam,
            gl + 1.0f, gm + 1.0f, gchan + 1.0f,
            ld*md*chd);

        // Determine the normalised angle
        FT angle = Po::arg(sum);

        // Take the complex exponent of the angle
        // and multiply by the sum of abs
        CT value;
        Po::sincos(angle, &value.y, &value.x);
        value.x *= abs_sum;
        value.y *= abs_sum;

        i = ((SRC*cdata.ntime + TIME)*cdata.na + ANT)*cdata.npolchan.extent_size() + POLCHAN;
        jones[i] = value;
        __syncthreads();
    }        
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
        // Get device address of GPU constant data
        cudaError_t error = cudaGetSymbolAddress((void **)&d_cdata, cdata);

        if(error != cudaSuccess) {
            printf("Cuda Error: %s\n", cudaGetErrorString(error));
        }
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

        // TODO. Does this actually allocate pinned memory?
        tf::AllocatorAttributes pinned_allocator;
        pinned_allocator.set_on_host(true);
        pinned_allocator.set_gpu_compatible(true);

        // Allocate memory for the constant data
        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DT_UINT8, cdata_shape, &cdata_tensor,
            pinned_allocator));

        // Cast raw bytes to the constant data structure type
        montblanc::ebeam::const_data * cdata_ptr = 
            reinterpret_cast<montblanc::ebeam::const_data *>(
                cdata_tensor.flat<uint8_t>().data());

        cdata_ptr->nsrc = nsrc;
        cdata_ptr->ntime = ntime;
        cdata_ptr->na = na;

        cdata_ptr->nchan.global_size = nchan;
        cdata_ptr->nchan.local_size = nchan;
        cdata_ptr->nchan.lower_extent = 0;
        cdata_ptr->nchan.upper_extent = nchan;

        cdata_ptr->npolchan.global_size = npolchan;
        cdata_ptr->npolchan.local_size = npolchan;
        cdata_ptr->npolchan.lower_extent = 0;
        cdata_ptr->npolchan.upper_extent = npolchan;

        cdata_ptr->beam_lw = in_E_beam.dim_size(0);
        cdata_ptr->beam_mh = in_E_beam.dim_size(1);
        cdata_ptr->beam_nud = in_E_beam.dim_size(2);

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

        FT parallactic_angle = in_parallactic_angle.tensor<FT, 0>()(0);
        FT beam_ll = in_beam_ll.tensor<FT, 0>()(0);
        FT beam_lm = in_beam_lm.tensor<FT, 0>()(0);
        FT beam_ul = in_beam_ul.tensor<FT, 0>()(0);
        FT beam_um = in_beam_um.tensor<FT, 0>()(0);

        rime_e_beam<Tr><<<grid, blocks, 0, stream>>>(
            lm, point_errors, antenna_scaling, E_beam,
            jones, parallactic_angle,
            beam_ll, beam_lm, beam_ul, beam_um);

    }
};

} // namespace ebeam {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #ifndef RIME_E_BEAM_OP_GPU_CUH_