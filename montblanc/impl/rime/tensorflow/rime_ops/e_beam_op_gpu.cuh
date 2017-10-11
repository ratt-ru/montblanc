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
// Mostly because of the beam frequency mapping index
// in shared memory
constexpr std::size_t BEAM_NUD_LIMIT = 128;

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
    const typename Traits::FT frequency,
    const int & beam_nud)
{
    using FT = typename Traits::FT;

    // OK these names and their usage are going to be confusing
    // Technically this function does an upper bound search,
    // the result of which is stored in the lower *variable*,
    // to save on registers.

    // This is different from the meaning of the lower and
    // upper bounds specified in the *arguments*.
    // So, at the end of this function,
    // upper is set to lower and lower is set to upper-1.
    lower = 0;
    upper = beam_nud-1;

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

    // Lower contains our upper bound result
    // Clamp it to at least 1
    upper = Policies::max(1, lower);
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
    const typename Traits::FT & weight,
    const int & beam_lw,
    const int & beam_mh,
    const int & beam_nud)
{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;

    int i = ((int(gl)*beam_mh + int(gm))*beam_nud +
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
    const typename Traits::FT * parallactic_angle_sin,
    const typename Traits::FT * parallactic_angle_cos,
    const typename Traits::FT * beam_freq_map,
    const typename Traits::CT * ebeam,
    typename Traits::CT * jones,
    const typename Traits::FT lower_l,
    const typename Traits::FT lower_m,
    const typename Traits::FT upper_l,
    const typename Traits::FT upper_m,
    int nsrc, int ntime, int na, int nchan, int npolchan,
    int beam_lw, int beam_mh, int beam_nud)
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

    if(TIME >= ntime || ANT >= na || POLCHAN >= npolchan)
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
    if(i < beam_nud)
    {
        shared.beam_freq_map[i] = beam_freq_map[i];
    }

    // Precompute l and m scaling factors in shared memory
    if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        shared.lscale = FT(beam_lw - 1) / (upper_l - lower_l);
        shared.mscale = FT(beam_mh - 1) / (upper_m - lower_m);
    }

    // Pointing errors vary by time, antenna and channel,
    if(ebeam_pol() == 0)
    {
        i = (TIME*na + ANT)*nchan + (POLCHAN >> 2);
        shared.pe[threadIdx.z][threadIdx.y][thread_chan()] = point_errors[i];
    }

    // Antenna scaling factors vary by antenna and channel, but not timestep
    if(threadIdx.z == 0 && ebeam_pol() == 0)
    {
        i = ANT*nchan + (POLCHAN >> 2);
        shared.as[threadIdx.y][thread_chan()] = antenna_scaling[i];
    }

    // Think this is needed so all beam_freq_map values are loaded
    __syncthreads();

    // Frequency vary by channel, but not timestep or antenna
    if(threadIdx.z == 0 && threadIdx.y == 0 && ebeam_pol() == 0)
    {
        // Get frequency and clamp to extents of the beam
        FT freq = frequency[POLCHAN >> 2];
        freq = Po::min(freq, shared.beam_freq_map[beam_nud-1]);
        freq = Po::max(freq, 0);

        int lower, upper;

        // Channel coordinate
        // channel grid position
        find_freq_bounds<Traits, Po>(lower, upper,
            shared.beam_freq_map, freq, beam_nud);

        // Snap to grid coordinate
        shared.gchan0[thread_chan()] = FT(lower);
        shared.gchan1[thread_chan()] = FT(upper);

        FT lower_freq = shared.beam_freq_map[lower];
        FT upper_freq = shared.beam_freq_map[upper];
        FT freq_diff = upper_freq - lower_freq;

        // Offset of snapped coordinate from grid position
        shared.chd[thread_chan()] = (freq - lower_freq)/freq_diff;
    }

    // Parallactic angles vary by time and antenna, but not channel
    if(threadIdx.x == 0)
    {
        i = TIME*na + ANT;
        shared.pa_sin[threadIdx.z][threadIdx.y] = parallactic_angle_sin[i];
        shared.pa_cos[threadIdx.z][threadIdx.y] = parallactic_angle_cos[i];
    }

    __syncthreads();

    for(int SRC=0; SRC < nsrc; ++SRC)
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
        l = shared.lscale * (l - lower_l);
        // clamp to grid edges
        l = Po::clamp(zero, l, beam_lw-1);
        // Snap to grid coordinate
        FT gl0 = Po::floor(l);
        FT gl1 = Po::min(gl0 + one, beam_lw-1);
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
        m = shared.mscale * (m - lower_m);
        // clamp to grid edges
        m = Po::clamp(zero, m, beam_mh-1);
        // Snap to grid position
        FT gm0 = Po::floor(m);
        FT gm1 = Po::min(gm0 + one, beam_mh-1);
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
            (one-ld)*(one-md)*(one-shared.chd[thread_chan()]),
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm0, shared.gchan0[thread_chan()],
            ld*(one-md)*(one-shared.chd[thread_chan()]),
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm1, shared.gchan0[thread_chan()],
            (one-ld)*md*(one-shared.chd[thread_chan()]),
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm1, shared.gchan0[thread_chan()],
            ld*md*(one-shared.chd[thread_chan()]),
            beam_lw, beam_mh, beam_nud);

        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm0, shared.gchan1[thread_chan()],
            (one-ld)*(one-md)*shared.chd[thread_chan()],
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm0, shared.gchan1[thread_chan()],
            ld*(one-md)*shared.chd[thread_chan()],
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl0, gm1, shared.gchan1[thread_chan()],
            (one-ld)*md*shared.chd[thread_chan()],
            beam_lw, beam_mh, beam_nud);
        trilinear_interpolate<Traits, Po>(pol_sum, abs_sum, ebeam,
            gl1, gm1, shared.gchan1[thread_chan()],
            ld*md*shared.chd[thread_chan()],
            beam_lw, beam_mh, beam_nud);

        // Normalise the angle and multiply in the absolute sum
        FT norm = Po::rsqrt(pol_sum.x*pol_sum.x + pol_sum.y*pol_sum.y);
        if(!::isfinite(norm))
            { norm = 1.0; }

        pol_sum.x *= norm * abs_sum;
        pol_sum.y *= norm * abs_sum;
        i = ((SRC*ntime + TIME)*na + ANT)*npolchan + POLCHAN;
        jones[i] = pol_sum;
    }
}

template <typename FT, typename CT>
class EBeam<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
    explicit EBeam(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_lm = context->input(0);
        const tf::Tensor & in_frequency = context->input(1);
        const tf::Tensor & in_point_errors = context->input(2);
        const tf::Tensor & in_antenna_scaling = context->input(3);
        const tf::Tensor & in_parallactic_angle_sin = context->input(4);
        const tf::Tensor & in_parallactic_angle_cos = context->input(5);
        const tf::Tensor & in_beam_extents = context->input(6);
        const tf::Tensor & in_beam_freq_map = context->input(7);
        const tf::Tensor & in_ebeam = context->input(8);

        // Extract problem dimensions
        int nsrc = in_lm.dim_size(0);
        int ntime = in_point_errors.dim_size(0);
        int na = in_point_errors.dim_size(1);
        int nchan = in_point_errors.dim_size(2);
        int npolchan = nchan*EBEAM_NPOL;
        int beam_lw = in_ebeam.dim_size(0);
        int beam_mh = in_ebeam.dim_size(1);
        int beam_nud = in_ebeam.dim_size(2);

        // Reason about our output shape
        // Create a pointer for the jones result
        tf::TensorShape jones_shape({nsrc, ntime, na, nchan, EBEAM_NPOL});
        tf::Tensor * jones_ptr = nullptr;

        // Allocate memory for the jones
        OP_REQUIRES_OK(context, context->allocate_output(
            0, jones_shape, &jones_ptr));

        if (jones_ptr->NumElements() == 0)
            { return; }

        OP_REQUIRES(context, beam_nud < BEAM_NUD_LIMIT,
            tf::errors::InvalidArgument("beam_nud must be less than '" +
                std::to_string(BEAM_NUD_LIMIT) + "' for the GPU beam."));

        // Extract beam extents
        auto beam_extents = in_beam_extents.tensor<FT, 1>();

        FT lower_l = beam_extents(0); // Lower l
        FT lower_m = beam_extents(1); // Lower m
        FT upper_l = beam_extents(3); // Upper l
        FT upper_m = beam_extents(4); // Upper m

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        typedef montblanc::kernel_traits<FT> Tr;
        typedef typename montblanc::ebeam::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(npolchan, na, ntime));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, npolchan, na, ntime));

        // Check that there are enough threads in the thread block
        // to properly load the beam frequency map into shared memory.
        OP_REQUIRES(context, blocks.x*blocks.y*blocks.z >= beam_nud,
            tf::errors::InvalidArgument("Not enough thread blocks to load "
                                                    "the beam frequency map"));

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
        auto parallactic_angle_sin = reinterpret_cast<
            const typename Tr::FT *>(
                in_parallactic_angle_sin.tensor<FT, 2>().data());
        auto parallactic_angle_cos = reinterpret_cast<
            const typename Tr::FT *>(
                in_parallactic_angle_cos.tensor<FT, 2>().data());
        auto beam_freq_map = reinterpret_cast<
            const typename Tr::FT *>(
                in_beam_freq_map.tensor<FT, 1>().data());
        auto ebeam = reinterpret_cast<
            const typename Tr::CT *>(
                in_ebeam.flat<CT>().data());

        rime_e_beam<Tr><<<grid, blocks, 0, stream>>>(
            lm, frequency, point_errors, antenna_scaling,
            parallactic_angle_sin, parallactic_angle_cos,
            beam_freq_map, ebeam, jones,
            lower_l, lower_m, upper_l, upper_m,
            nsrc, ntime, na, nchan, npolchan,
            beam_lw, beam_mh, beam_nud);

    }
};

} // namespace ebeam {
} // namespace montblanc {

#endif // #if GOOGLE_CUDA

#endif // #ifndef RIME_E_BEAM_OP_GPU_CUH_
