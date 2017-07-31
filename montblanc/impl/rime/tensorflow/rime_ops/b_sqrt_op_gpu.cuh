#ifndef RIME_B_SQRT_OP_GPU_H_
#define RIME_B_SQRT_OP_GPU_H_

#if GOOGLE_CUDA

#include "b_sqrt_op.h"
#include <montblanc/abstraction.cuh>
#include <montblanc/brightness.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace montblanc {
namespace bsqrt {

// Traits class defined by float and complex types
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

constexpr int BSQRT_NPOL = 4;

template <typename Traits> __device__ __forceinline__
int bsqrt_pol()
    { return threadIdx.x & 0x3; }

template <typename Traits>
__global__ void rime_b_sqrt(
    const typename Traits::stokes_type * stokes,
    const typename Traits::alpha_type * alpha,
    const typename Traits::frequency_type * frequency,
    const typename Traits::frequency_type * ref_freq,
    typename Traits::visibility_type * B_sqrt,
    typename Traits::sgn_brightness_type * sgn_brightness,
    bool linear,
    int nsrc, int ntime, int nchan)
{
    // Simpler float and complex types
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using ST = typename Traits::stokes_type;
    using VT = typename Traits::visibility_type;

    constexpr FT zero = 0.0;
    constexpr FT one = 1.0;

    typedef typename montblanc::kernel_policies<FT> Po;
    typedef typename montblanc::bsqrt::LaunchTraits<FT> LTr;

    int CHAN = blockIdx.x*blockDim.x + threadIdx.x;
    int TIME = blockIdx.y*blockDim.y + threadIdx.y;
    int SRC = blockIdx.z*blockDim.z + threadIdx.z;

    if(SRC >= nsrc || TIME >= ntime || CHAN >= nchan)
        { return; }

    __shared__ FT src_rfreq[LTr::BLOCKDIMZ];
    __shared__ FT freq[LTr::BLOCKDIMX];

    // Varies by channel
    if(threadIdx.z == 0 && threadIdx.y == 0)
    {
        freq[threadIdx.x] = frequency[CHAN];
    }

    // Varies by source
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        src_rfreq[threadIdx.z] = ref_freq[SRC];
    }

    __syncthreads();

    // Calculate power term
    int i = SRC*ntime + TIME;
    FT power = Po::pow(freq[threadIdx.x]/src_rfreq[threadIdx.z], alpha[i]);

    // Read in stokes parameters (IQUV)
    ST _stokes = stokes[i];
    FT & I = linear ? _stokes.x : _stokes.x;
    FT & Q = linear ? _stokes.y : _stokes.w;
    FT & U = linear ? _stokes.z : _stokes.y;
    FT & V = linear ? _stokes.w : _stokes.z;

    I *= power;
    Q *= power;
    U *= power;
    V *= power;

    // sgn variable, used to indicate whether
    // brightness matrix is negative, zero or positive
    FT IQ = I + Q;
    FT sgn = (zero < IQ) - (IQ < zero);

    // Indicate that we inverted the sgn of the brightness
    // matrix to obtain the cholesky decomposition
    i = SRC*ntime + TIME;
    if(CHAN == 0)
        { sgn_brightness[i] = sgn; }

    // I *= sgn;
    // Q *= sgn;
    U *= sgn;
    V *= sgn;
    IQ *= sgn;

    // Create the cholesky decomposition of the brightness matrix
    // L00 = sqrt(I+Q)
    // L01 = 0
    // L10 = (U+iV)/sqrt(I+Q)
    // L11 = sqrt((I**2 - Q**2 - U**2 - V**2)/(I+Q))

    // Use the YY and XY correlations as scratch space
    VT B;
    // L00 = sqrt(I+Q)
    B.XY.x = IQ; B.XY.y = zero;
    B.XX = Po::sqrt(B.XY);
    // Store L00 as a divisor of L10
    B.XY = B.XX;

    // Gracefully handle zero matrices
    if(IQ == zero)
    {
        B.XY.x = one; B.XY.y = zero;
        IQ = one;
    }

    // L10 = (U-iV)/sqrt(I+Q)
    B.YY.x = U; B.YY.y = -V;
    B.YX.x = B.YY.x*B.XY.x + B.YY.y*B.XY.y;
    B.YX.y = B.YY.y*B.XY.x - B.YY.x*B.XY.y;
    B.YY.x = one/Po::abs_squared(B.XY);
    B.YX.x *= B.YY.x;
    B.YX.y *= B.YY.x;

    // L11 = sqrt((I**2 - Q**2 - U**2 - V**2)/(I+Q))
    B.XY.x = (I*I - Q*Q - U*U - V*V)/IQ;
    B.XY.y = zero;
    B.YY = Po::sqrt(B.XY);

    // L01 = 0
    B.XY.x = zero; B.XY.y = zero;

    // Write out the cholesky decomposition
    B_sqrt[i*nchan + CHAN] = B;
}

template <typename FT, typename CT>
class BSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
public:
private:
    std::string polarisation_type;

public:
    explicit BSqrt(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("polarisation_type",
                                                 &polarisation_type));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_stokes = context->input(0);
        const tf::Tensor & in_alpha = context->input(1);
        const tf::Tensor & in_frequency = context->input(2);
        const tf::Tensor & in_ref_freq = context->input(3);

        // Extract problem dimensions
        int nsrc = in_stokes.dim_size(0);
        int ntime = in_stokes.dim_size(1);
        int nchan = in_frequency.dim_size(0);

        bool linear = (polarisation_type == "linear");

        // Reason about the shape of the b_sqrt tensor and
        // create a pointer to it
        tf::TensorShape b_sqrt_shape({nsrc, ntime, nchan, 4});
        tf::Tensor * b_sqrt_ptr = nullptr;

        // Allocate memory for the b_sqrt
        OP_REQUIRES_OK(context, context->allocate_output(
            0, b_sqrt_shape, &b_sqrt_ptr));

        if (b_sqrt_ptr->NumElements() == 0)
            { return; }

        // Reason about shape of the invert tensor
        // and create a pointer to it
        tf::TensorShape invert_shape({nsrc, ntime});
        tf::Tensor * invert_ptr = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(
            1, invert_shape, &invert_ptr));

        // Cast input into CUDA types defined within the Traits class
        typedef montblanc::kernel_traits<FT> Tr;
        typedef montblanc::bsqrt::LaunchTraits<FT> LTr;

        // Set up our kernel dimensions
        dim3 blocks(LTr::block_size(nchan, ntime, nsrc));
        dim3 grid(montblanc::grid_from_thread_block(
            blocks, nchan, ntime, nsrc));

        // Get the device pointers of our GPU memory arrays
        auto stokes = reinterpret_cast<const typename Tr::stokes_type *>(
            in_stokes.flat<FT>().data());
        auto alpha = reinterpret_cast<const typename Tr::alpha_type *>(
            in_alpha.flat<FT>().data());
        auto frequency = reinterpret_cast<const typename Tr::frequency_type *>(
            in_frequency.flat<FT>().data());
        auto ref_freq = reinterpret_cast<const typename Tr::frequency_type *>(
            in_ref_freq.flat<FT>().data());
        auto b_sqrt = reinterpret_cast<typename Tr::visibility_type *>(
            b_sqrt_ptr->flat<CT>().data());
        auto sgn_brightness = reinterpret_cast<typename Tr::sgn_brightness_type *>(
            invert_ptr->flat<tf::int8>().data());

        const auto & stream = context->eigen_device<GPUDevice>().stream();

        rime_b_sqrt<Tr> <<<grid, blocks, 0, stream>>>(stokes,
            alpha, frequency, ref_freq,
            b_sqrt, sgn_brightness,
            linear,
            nsrc, ntime, nchan);
    }
};

} // namespace sqrt {
} // namespace montblanc {

#endif

#endif // #ifndef RIME_B_SQRT_OP_GPU_H_