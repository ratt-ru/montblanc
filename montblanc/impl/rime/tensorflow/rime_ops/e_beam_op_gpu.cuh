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
    static constexpr int BLOCKDIMY = 8;
    static constexpr int BLOCKDIMZ = 4;

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
    static constexpr int BLOCKDIMY = 8;
    static constexpr int BLOCKDIMZ = 4;

    static dim3 block_size(int X, int Y, int Z)
    {
        return montblanc::shrink_small_dims(
            dim3(BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ),
            X, Y, Z);
    }
};


// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// Limit the BEAM_NUD dimension we're prepared to handle
// Mostly because of the second index beam
// frequency mapping in shared memory
constexpr std::size_t BEAM_NUD_LIMIT = 128;

// Constant GPU memory. Note that declaring constant memory
// with templates is a bit tricky, so we declare it as
// double (the largest type it should take) and cast it using
// the holder struct below
__constant__ montblanc::ebeam::const_data<double> beam_constant;

// Helper class for casting constant data to appropriate type
template <typename T>
struct holder
{
    using bc = typename montblanc::ebeam::const_data<T>;
    using bc_ptr = typename montblanc::ebeam::const_data<T> *;

    static __device__ __forceinline__ bc_ptr ptr () ;
};

// Specialise for float
template <> __device__ __forceinline__ holder<float>::bc_ptr
holder<float>::ptr()
    { return reinterpret_cast<holder<float>::bc_ptr>(&beam_constant); }

// Specialise for double
template <> __device__ __forceinline__ holder<double>::bc_ptr
holder<double>::ptr()
    { return reinterpret_cast<holder<double>::bc_ptr>(&beam_constant); }

#define cdata holder<FT>::ptr()

// Get the current polarisation from the thread ID
__device__ __forceinline__ int ebeam_pol()
    { return threadIdx.x & 0x3; }

// Get the current channel from the thread ID
__device__ __forceinline__ int thread_chan()
    { return threadIdx.x >> 2; }

template <typename Traits, typename Policies>
__device__ __forceinline__
void find_freq_bounds(int & lower, int & upper,
    const typename Traits::FT * beam_freq_map,
    const typename Traits::FT frequency)
{
    using FT = typename Traits::FT;

    lower = 0;
    upper = cdata->beam_nud;

    // Warp divergence here, unlikely
    // to be a big deal though
    while(lower < upper)
    {
        int i = (lower + upper)/2;

        if(frequency < beam_freq_map[i])
            { upper = i; }
        else
            { lower = i + 1; }
    }

    upper = lower;
    lower = upper - 1;
}

template <typename Traits, typename Policies>
__device__ __forceinline__
void trilinear_interpolate(
    typename Traits::CT & pol_sum,
    typename Traits::FT & abs_sum,
    const typename Traits::CT * ebeam,
    const typename Traits::FT gl,
    const typename Traits::FT gm,
    const typename Traits::FT gchan,
    const typename Traits::FT & weight)
{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;

    int i = ((int(gl)*cdata->beam_mh + int(gm))*cdata->beam_nud +
        int(gchan))*EBEAM_NPOL + ebeam_pol();

    // Perhaps unnecessary as long as BLOCKDIMX is 32
    CT data = cub::ThreadLoad<cub::LOAD_LDG>(ebeam + i);
    pol_sum.x += weight*data.x;
    pol_sum.y += weight*data.y;
    abs_sum += weight*Policies::abs(data);
}

template <typename Traits>
__global__ void rime_e_beam(
    const typename Traits::lm_type * lm,
    const typename Traits::frequency_type * frequency,
    const typename Traits::point_error_type * point_errors,
    const typename Traits::antenna_scale_type * antenna_scaling,
    const typename Traits::FT * parallactic_angle,
    const typename Traits::FT * beam_freq_map,
    const typename Traits::CT * ebeam,
    typename Traits::CT * jones)
{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;

    using lm_type = typename Traits::lm_type;
    using point_error_type = typename Traits::point_error_type;
    using antenna_scale_type = typename Traits::antenna_scale_type;

    using Po = typename montblanc::kernel_policies<FT>;
    using LTr = typename montblanc::ebeam::LaunchTraits<FT>;

    int POLCHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int ANT = blockIdx.y*blockDim.y + threadIdx.y;
    int TIME = blockIdx.z*blockDim.z + threadIdx.z;
    constexpr int BLOCKCHANS = LTr::BLOCKDIMX >> 2;
    constexpr FT zero = 0.0f;
    constexpr FT one = 1.0f;

    if(TIME >= cdata->ntime || ANT >= cdata->na || POLCHAN >= cdata->npolchan)
        { return; }

    __shared__ struct {
        FT beam_freq_map[BEAM_NUD_LIMIT];
        FT lscale;             // l axis scaling factor
        FT mscale;             // m axis scaling factor
        FT pa_sin[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];  // sin of parallactic angle
        FT pa_cos[LTr::BLOCKDIMZ][LTr::BLOCKDIMY];  // cos of parallactic angle
        FT gchan0[BLOCKCHANS];  // channel grid position (snapped)
        FT gchan1[BLOCKCHANS];  // channel grid position (snapped)
        FT chd[BLOCKCHANS];    // difference between gchan0 and actual grid position
        // pointing errors
        point_error_type pe[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][BLOCKCHANS];
        // antenna scaling
        antenna_scale_type as[LTr::BLOCKDIMY][BLOCKCHANS];
    } shared;

    int i;


    // 3D thread ID
    i = threadIdx.z*blockDim.x*blockDim.y
        + threadIdx.y*blockDim.x
        + threadIdx.x;

    // Load in the beam frequency mapping
    if(i < cdata->beam_nud)
    {
        shared.beam_freq_map[i] = beam_freq_map[i];
    }

    // Precompute l and m scaling factors in shared memory
    if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        shared.lscale = FT(cdata->beam_lw - 1) / (cdata->ul - cdata->ll);
        shared.mscale = FT(cdata->beam_mh - 1) / (cdata->um - cdata->lm);
    }

    // Pointing errors vary by time, antenna and channel,
    if(ebeam_pol() == 0)
    {
        i = (TIME*cdata->na + ANT)*cdata->nchan + (POLCHAN >> 2);
        shared.pe[threadIdx.z][threadIdx.y][thread_chan()] = point_errors[i];
    }

    // Antenna scaling factors vary by antenna and channel, but not timestep
    if(threadIdx.z == 0 && ebeam_pol() == 0)
    {
        i = ANT*cdata->nchan + (POLCHAN >> 2);
        shared.as[threadIdx.y][thread_chan()] = antenna_scaling[i];
    }

    // Think this is needed so all beam_freq_map values are loaded
    __syncthreads();

    // Frequency vary by channel, but not timestep or antenna
    if(threadIdx.z == 0 && threadIdx.y == 0 && ebeam_pol() == 0)
    {
        // Channel coordinate
        // channel grid position
        FT freq = frequency[POLCHAN >> 2];
        int lower, upper;

        find_freq_bounds<Traits, Po>(lower, upper, beam_freq_map, freq);

        // Snap to grid coordinate
        shared.gchan0[thread_chan()] = FT(lower);
        shared.gchan1[thread_chan()] = FT(upper);

        FT lower_freq = beam_freq_map[lower];
        FT upper_freq = beam_freq_map[upper];
        FT freq_diff = upper_freq - lower_freq;

        // Offset of snapped coordinate from grid position
        shared.chd[thread_chan()] = (freq - lower_freq)/freq_diff;
    }

    // Parallactic angles vary by time and antenna, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*cdata->na + ANT;
        FT parangle = parallactic_angle[i];
        Po::sincos(parangle,
            &shared.pa_sin[threadIdx.z][threadIdx.y],
            &shared.pa_cos[threadIdx.z][threadIdx.y]);
    }

    __syncthreads();

    for(int SRC=0; SRC < cdata->nsrc; ++SRC)
    {
        lm_type rlm = lm[SRC];

       // L coordinate
        // Rotate
        FT l = rlm.x*shared.pa_cos[threadIdx.z][threadIdx.y] -
            rlm.y*shared.pa_sin[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        l += shared.pe[threadIdx.z][threadIdx.y][thread_chan()].x;
        // Scale by antenna scaling factors
        l *= shared.as[threadIdx.y][thread_chan()].x;
        // l grid position
        l = shared.lscale * (l - cdata->ll);
        // clamp to grid edges
        l = Po::clamp(zero, l, cdata->beam_lw-1);
        // Snap to grid coordinate
        FT gl0 = Po::floor(l);
        FT gl1 = Po::min(gl0 + one, cdata->beam_lw-1);
        // Offset of snapped coordinate from grid position
        FT ld = l - gl0;

        // M coordinate
        // rotate
        FT m = rlm.x*shared.pa_sin[threadIdx.z][threadIdx.y] +
            rlm.y*shared.pa_cos[threadIdx.z][threadIdx.y];
        // Add the pointing errors for this antenna.
        m += shared.pe[threadIdx.z][threadIdx.y][thread_chan()].y;
        // Scale by antenna scaling factors
        m *= shared.as[threadIdx.y][thread_chan()].y;
        // m grid position
        m = shared.mscale * (m - cdata->lm);
        // clamp to grid edges
        m = Po::clamp(zero, m, cdata->beam_mh-1);
        // Snap to grid position
        FT gm0 = Po::floor(m);
        FT gm1 = Po::min(gm0 + one, cdata->beam_mh-1);
        // Offset of snapped coordinate from grid position
        FT md = m - gm0;

        CT pol_sum = Po::make_ct(zero, zero);
        FT abs_sum = FT(zero);
        // A simplified trilinear weighting is used here. Given
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
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm0, shared.gchan0[thread_chan()],
            (one-ld)*(one-md)*(one-shared.chd[thread_chan()]));
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm0, shared.gchan0[thread_chan()],
            ld*(one-md)*(one-shared.chd[thread_chan()]));
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm1, shared.gchan0[thread_chan()],
            (one-ld)*md*(one-shared.chd[thread_chan()]));
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm1, shared.gchan0[thread_chan()],
            ld*md*(one-shared.chd[thread_chan()]));

        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm0, shared.gchan1[thread_chan()],
            (one-ld)*(one-md)*shared.chd[thread_chan()]);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm0, shared.gchan1[thread_chan()],
            ld*(one-md)*shared.chd[thread_chan()]);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm1, shared.gchan1[thread_chan()],
            (one-ld)*md*shared.chd[thread_chan()]);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm1, shared.gchan1[thread_chan()],
            ld*md*shared.chd[thread_chan()]);

        // Normalise the angle and multiply in the absolute sum
        FT norm = Po::rsqrt(pol_sum.x*pol_sum.x + pol_sum.y*pol_sum.y);
        if(!::isfinite(norm))
            { norm = 1.0; }

        pol_sum.x *= norm * abs_sum;
        pol_sum.y *= norm * abs_sum;
        i = ((SRC*cdata->ntime + TIME)*cdata->na + ANT)*cdata->npolchan + POLCHAN;
        jones[i] = pol_sum;
    }
}

template <typename FT, typename CT>
class EBeam<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    // Pointer to constant memory on the device
    montblanc::ebeam::const_data<FT> * d_cdata;

public:
    explicit EBeam(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        // Get device address of GPU constant data
        cudaError_t error = cudaGetSymbolAddress((void **)&d_cdata, beam_constant);

        if(error != cudaSuccess) {
            printf("Cuda Error: %s\n", cudaGetErrorString(error));
        }
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_frequency = context->input(1);
        const tf::Tensor & in_point_errors = context->input(2);
        const tf::Tensor & in_antenna_scaling = context->input(3);
        const tf::Tensor & in_parallactic_angle = context->input(4);
        const tf::Tensor & in_beam_extents = context->input(5);
        const tf::Tensor & in_beam_freq_map = context->input(6);
        const tf::Tensor & in_ebeam = context->input(7);

        OP_REQUIRES(context, in_lm.dims() == 2 && in_lm.dim_size(1) == 2,
            tf::errors::InvalidArgument("lm should be of shape (nsrc, 2)"))

        OP_REQUIRES(context, in_frequency.dims() == 1,
            tf::errors::InvalidArgument("frequency should be of shape (nchan,)"))

        OP_REQUIRES(context, in_point_errors.dims() == 4
            && in_point_errors.dim_size(3) == 2,
            tf::errors::InvalidArgument("point_errors should be of shape "
                                        "(ntime, na, nchan, 2)"))

        OP_REQUIRES(context, in_antenna_scaling.dims() == 3
            && in_antenna_scaling.dim_size(2) == 2,
            tf::errors::InvalidArgument("antenna_scaling should be of shape "
                                        "(na, nchan, 2)"))

        OP_REQUIRES(context, in_ebeam.dims() == 4
            && in_ebeam.dim_size(3) == 4,
            tf::errors::InvalidArgument("E_Beam should be of shape "
                                        "(beam_lw, beam_mh, beam_nud, 4)"))

        OP_REQUIRES(context, in_parallactic_angle.dims() == 2,
            tf::errors::InvalidArgument("parallactic_angle should be of shape "
                                        "(ntime, na)"))

        OP_REQUIRES(context, in_beam_extents.dims() == 1
            && in_beam_extents.dim_size(0) == 6,
            tf::errors::InvalidArgument("beam_extents should be of shape "
                                        "(6,)"))

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

        int cdata_size = sizeof(montblanc::ebeam::const_data<FT>);

        tf::TensorShape cdata_shape({cdata_size});
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
        montblanc::ebeam::const_data<FT> * cdata_ptr =
            reinterpret_cast<montblanc::ebeam::const_data<FT> *>(
                cdata_tensor.flat<uint8_t>().data());

        cdata_ptr->nsrc = nsrc;
        cdata_ptr->ntime = ntime;
        cdata_ptr->na = na;
        cdata_ptr->nchan = nchan;
        cdata_ptr->npol = EBEAM_NPOL;
        cdata_ptr->npolchan = npolchan;

        cdata_ptr->beam_lw = in_ebeam.dim_size(0);
        cdata_ptr->beam_mh = in_ebeam.dim_size(1);
        cdata_ptr->beam_nud = in_ebeam.dim_size(2);

        OP_REQUIRES(context, cdata_ptr->beam_nud < BEAM_NUD_LIMIT,
            tf::errors::InvalidArgument("beam_nud must be less than '" +
                std::to_string(BEAM_NUD_LIMIT) + "' for the GPU beam."));

        // Extract beam extents
        auto beam_extents = in_beam_extents.tensor<FT, 1>();

        cdata_ptr->ll = beam_extents(0); // Lower l
        cdata_ptr->lm = beam_extents(1); // Lower m
        cdata_ptr->lf = beam_extents(2); // Lower frequency
        cdata_ptr->ul = beam_extents(3); // Upper l
        cdata_ptr->um = beam_extents(4); // Upper m
        cdata_ptr->uf = beam_extents(5); // Upper frequency

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        // Enqueue a copy of constant data to the device
        cudaMemcpyAsync(d_cdata, cdata_ptr, cdata_size,
           cudaMemcpyHostToDevice, stream);

        typedef montblanc::kernel_traits<FT> Tr;
        typedef typename montblanc::ebeam::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(npolchan, na, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, npolchan, na, ntime));

        // Cast to the cuda types expected by the kernel
        auto lm = reinterpret_cast<
            const typename Tr::lm_type *>(
                in_lm.flat<FT>().data());
        auto frequency = reinterpret_cast<
            const typename Tr::frequency_type *>(
                in_frequency.flat<FT>().data());
        auto point_errors = reinterpret_cast<
            const typename Tr::point_error_type *>(
                in_point_errors.flat<FT>().data());
        auto antenna_scaling = reinterpret_cast<
            const typename Tr::antenna_scale_type *>(
                in_antenna_scaling.flat<FT>().data());
        auto jones = reinterpret_cast<typename Tr::CT *>(
                jones_ptr->flat<CT>().data());
        auto parallactic_angle = reinterpret_cast<
            const typename Tr::FT *>(
                in_parallactic_angle.tensor<FT, 2>().data());
        auto beam_freq_map = reinterpret_cast<
            const typename Tr::FT *>(
                in_beam_freq_map.tensor<FT, 1>().data());
        auto ebeam = reinterpret_cast<
            const typename Tr::CT *>(
                in_ebeam.flat<CT>().data());

        rime_e_beam<Tr><<<grid, blocks, 0, stream>>>(
            lm, frequency, point_errors, antenna_scaling,
            parallactic_angle, beam_freq_map, ebeam, jones);

    }
};

} // namespace ebeam {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #ifndef RIME_E_BEAM_OP_GPU_CUH_