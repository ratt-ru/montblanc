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

template <typename T>
__device__  __forceinline__ void device_swap(T & a, T & b)
{
    T c(a); a=b; b=c;
}

// CUDA kernel outline
template <typename Traits>
__global__ void rime_create_antenna_jones(
    const typename Traits::CT * bsqrt,
    const typename Traits::CT * complex_phase,
    const typename Traits::CT * feed_rotation,
    const typename Traits::CT * ddes,
    typename Traits::CT * ant_jones,
    int nsrc, int ntime, int na, int nchan, int npol)
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;
    using Po = typename montblanc::kernel_policies<FT>;

    int polchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = polchan / npol;
    int pol = polchan & (npol-1);
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;
    int npolchan = nchan*npol;

    if(time > ntime || ant >= na || polchan > npolchan)
        { return; }

    int i;

    __shared__ struct {
        CT fr[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][CREATE_ANTENNA_JONES_NPOL];
    } shared;

    // Feed rotation varies by time, antenna and polarisation
    // Polarisation is baked into the X dimension, so use the
    // first npol threads to load polarisation info
    if(feed_rotation != nullptr && threadIdx.x < npol)
    {
        i = (time*na + ant)*npol + pol;
        shared.fr[threadIdx.z][threadIdx.y][threadIdx.x] = feed_rotation[i];
    }

    __syncthreads();

    using montblanc::jones_multiply_4x4_in_place;
    using montblanc::complex_multiply_in_place;

    for(int src=0; src < nsrc; ++src)
    {
        CT buf[2];
        int a = 0, in = 1;
        bool initialised = 0;

        if(bsqrt != nullptr)
        {
            // Load and multiply the brightness square root
            i = (src*ntime + time)*npolchan + polchan;
            buf[in] = bsqrt[i];
            if(initialised)
                { jones_multiply_4x4_in_place<FT>(buf[in], buf[a]); }
            else
                { initialised = true; }
            device_swap(a, in);
        }

        if(complex_phase != nullptr)
        {
            // Load and multiply the complex phase
            i = ((src*ntime + time)*na + ant)*nchan + chan;
            buf[in] = complex_phase[i];
            if(initialised)
                { complex_multiply_in_place<FT>(buf[in], buf[a]); }
            else
                { initialised = true; }
            device_swap(a, in);
        }

        if(feed_rotation != nullptr)
        {
            // Load and multiply the feed rotation
            buf[in] = shared.fr[threadIdx.z][threadIdx.y][pol];
            if(initialised)
                { jones_multiply_4x4_in_place<FT>(buf[in], buf[a]); }
            else
                { initialised = true; }
            device_swap(a, in);
        }

        i = ((src*ntime + time)*na + ant)*npolchan + polchan;

        if(ddes != nullptr)
        {
            // Load and multiply the ddes
            buf[in] = ddes[i];
            if(initialised)
                { jones_multiply_4x4_in_place<FT>(buf[in], buf[a]); }
            else
                { initialised = true; }
            device_swap(a, in);
        }

        // If still uninitialised, set to jones identity
        if(!initialised)
            { buf[a] = montblanc::jones_identity<FT>(); }

        // Output final per antenna value
        ant_jones[i] = buf[a];
    }
}

// Specialise the CreateAntennaJones op for GPUs
template <typename FT, typename CT>
class CreateAntennaJones<GPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    bool have_bsqrt;
    bool have_complex_phase;
    bool have_feed_rotation;
    bool have_ddes;

public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context),
        have_bsqrt(false),
        have_complex_phase(false),
        have_feed_rotation(false),
        have_ddes(false)
    {
        OP_REQUIRES_OK(context, context->GetAttr("have_bsqrt",
                                                 &have_bsqrt));
        OP_REQUIRES_OK(context, context->GetAttr("have_complex_phase",
                                                 &have_complex_phase));
        OP_REQUIRES_OK(context, context->GetAttr("have_feed_rotation",
                                                 &have_feed_rotation));
        OP_REQUIRES_OK(context, context->GetAttr("have_ddes",
                                                 &have_ddes));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_bsqrt = context->input(0);
        const tf::Tensor & in_complex_phase = context->input(1);
        const tf::Tensor & in_feed_rotation = context->input(2);
        const tf::Tensor & in_ddes = context->input(3);

        int nsrc = -1, ntime = -1, na = -1, nchan = -1, npol = -1;

        auto update_dim = [](int & old_size,
                            const tf::Tensor & tensor,
                            int dim) -> tf::Status
        {
            auto new_size = tensor.dim_size(dim);

            if(old_size == -1)
            {
                old_size = new_size;
            }
            else if(old_size != new_size)
            {
                return tf::Status(tf::errors::InvalidArgument(
                        "Previously set dimension size '",  old_size,
                        "' does not equal new size '", new_size, "'"));
            }

            return tf::Status::OK();
        };

        if(have_bsqrt)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_bsqrt, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_bsqrt, 1));
            OP_REQUIRES_OK(context, update_dim(nchan, in_bsqrt, 2));
            OP_REQUIRES_OK(context, update_dim(npol, in_bsqrt, 3));
        }

        if(have_complex_phase)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_complex_phase, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_complex_phase, 1));
            OP_REQUIRES_OK(context, update_dim(na, in_complex_phase, 2));
            OP_REQUIRES_OK(context, update_dim(nchan, in_complex_phase, 3));
        }

        if(have_feed_rotation)
        {
            OP_REQUIRES_OK(context, update_dim(ntime, in_feed_rotation, 0));
            OP_REQUIRES_OK(context, update_dim(na, in_feed_rotation, 1));
        }

        if(have_ddes)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_ddes, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_ddes, 1));
            OP_REQUIRES_OK(context, update_dim(na, in_ddes, 2));
            OP_REQUIRES_OK(context, update_dim(nchan, in_ddes, 3));
            OP_REQUIRES_OK(context, update_dim(npol, in_ddes, 4));
        }

        //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, npol == CREATE_ANTENNA_JONES_NPOL,
            tf::errors::InvalidArgument("Number of polarisations '",
                npol, "' does not equal '", CREATE_ANTENNA_JONES_NPOL, "'."));

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, npol});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));


        using LTr = LaunchTraits<FT>;
        using Tr =  montblanc::kernel_traits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            npol*nchan, na, ntime);
        dim3 grid(montblanc::grid_from_thread_block(
            block, npol*nchan, na, ntime));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Get pointers to flattened tensor data buffers
        auto bsqrt = reinterpret_cast<const typename Tr::CT *>(
            in_bsqrt.flat<CT>().data());
        auto complex_phase = reinterpret_cast<const typename Tr::CT *>(
            in_complex_phase.flat<CT>().data());
        auto feed_rotation = reinterpret_cast<const typename Tr::CT *>(
            in_feed_rotation.flat<CT>().data());
        auto ddes = reinterpret_cast<const typename Tr::CT *>(
            in_ddes.flat<CT>().data());
        auto ant_jones = reinterpret_cast<typename Tr::CT *>(
            ant_jones_ptr->flat<CT>().data());

        // Call the rime_create_antenna_jones CUDA kernel
        rime_create_antenna_jones<Tr><<<grid, block, 0, device.stream()>>>(
            have_bsqrt ? bsqrt : nullptr,
            have_complex_phase ? complex_phase : nullptr,
            have_feed_rotation ? feed_rotation : nullptr,
            have_ddes ? ddes : nullptr,
            ant_jones,
            nsrc, ntime, na, nchan, npol);
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
