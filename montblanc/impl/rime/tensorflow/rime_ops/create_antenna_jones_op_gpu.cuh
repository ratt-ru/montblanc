#if GOOGLE_CUDA

#ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH
#define RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#include <montblanc/abstraction.cuh>
#include <montblanc/jones.cuh>

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "create_antenna_jones_op.h"
#include "shapes.h"

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
    int nsrc, int ntime, int na, int nchan, int ncorr)
{
    using FT = typename Traits::FT;
    using CT = typename Traits::CT;
    using LTr = LaunchTraits<FT>;
    using Po = typename montblanc::kernel_policies<FT>;

    int corrchan = blockIdx.x*blockDim.x + threadIdx.x;
    int chan = corrchan / ncorr;
    int corr = corrchan & (ncorr-1);
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;
    int ncorrchan = nchan*ncorr;

    if(time > ntime || ant >= na || corrchan > ncorrchan)
        { return; }

    int i;

    __shared__ struct {
        CT fr[LTr::BLOCKDIMZ][LTr::BLOCKDIMY][CREATE_ANTENNA_JONES_NCORR];
    } shared;

    // Feed rotation varies by time, antenna and polarisation
    // Polarisation is baked into the X dimension, so use the
    // first ncorr threads to load polarisation info
    if(feed_rotation != nullptr && threadIdx.x < ncorr)
    {
        i = (time*na + ant)*ncorr + corr;
        shared.fr[threadIdx.z][threadIdx.y][threadIdx.x] = feed_rotation[i];
    }

    __syncthreads();

    using montblanc::jones_multiply_4x4_in_place;
    using montblanc::complex_multiply_in_place;

    for(int src=0; src < nsrc; ++src)
    {
        CT buf[2];
        int a = 0, in = 1;
        bool initialised = false;

        if(bsqrt != nullptr)
        {
            // Load and multiply the brightness square root
            i = (src*ntime + time)*ncorrchan + corrchan;
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
            buf[in] = shared.fr[threadIdx.z][threadIdx.y][corr];
            if(initialised)
                { jones_multiply_4x4_in_place<FT>(buf[in], buf[a]); }
            else
                { initialised = true; }
            device_swap(a, in);
        }

        i = ((src*ntime + time)*na + ant)*ncorrchan + corrchan;

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
    std::string bsqrt_schema;
    std::string complex_phase_schema;
    std::string feed_rotation_schema;
    std::string ddes_schema;

public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        OP_REQUIRES_OK(context, context->GetAttr("bsqrt_schema",
                                                 &bsqrt_schema));
        OP_REQUIRES_OK(context, context->GetAttr("complex_phase_schema",
                                                 &complex_phase_schema));
        OP_REQUIRES_OK(context, context->GetAttr("feed_rotation_schema",
                                                 &feed_rotation_schema));
        OP_REQUIRES_OK(context, context->GetAttr("ddes_schema",
                                                 &ddes_schema));

        int have;

        OP_REQUIRES_OK(context, context->GetAttr("have_bsqrt", &have));
        OP_REQUIRES(context, have <= 1,
                    InvalidArgument("have_bsqrt > 1"));

        OP_REQUIRES_OK(context, context->GetAttr("have_complex_phase", &have));
        OP_REQUIRES(context, have <= 1,
                    InvalidArgument("have_complex_phase > 1"));

        OP_REQUIRES_OK(context, context->GetAttr("have_feed_rotation", &have));
        OP_REQUIRES(context, have <= 1,
                    InvalidArgument("have_feed_rotation > 1"));

        OP_REQUIRES_OK(context, context->GetAttr("have_ddes", &have));
        OP_REQUIRES(context, have <= 1,
                    InvalidArgument("have_ddes > 1"));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        ComputeInputDimSizes input_dim_sizes;

        tf::OpInputList bsqrt_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "bsqrt",
                                      bsqrt_schema,
                                      input_dim_sizes,
                                      bsqrt_list));

        tf::OpInputList complex_phase_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "complex_phase",
                                      complex_phase_schema,
                                      input_dim_sizes,
                                      complex_phase_list));

        tf::OpInputList feed_rotation_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "feed_rotation",
                                      feed_rotation_schema,
                                      input_dim_sizes,
                                      feed_rotation_list));

        tf::OpInputList ddes_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "ddes",
                                      ddes_schema,
                                      input_dim_sizes,
                                      ddes_list));

        ComputeDimSizes dim_sizes;
        OP_REQUIRES_OK(context, merge_input_dims(input_dim_sizes, dim_sizes));

        ComputeDimSizes::const_iterator it;
        ComputeDimSizes::const_iterator end = dim_sizes.end();

        OP_REQUIRES(context, (it = dim_sizes.find("source")) != end,
                    InvalidArgument("No source dimension found"));
        int nsrc = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("time")) != end,
                    InvalidArgument("No time dimension found"));
        int ntime = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("ant")) != end,
                    InvalidArgument("No ant dimension found"));
        int na = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("chan")) != end,
                    InvalidArgument("No chan dimension found"));
        int nchan = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("corr")) != end,
                    InvalidArgument("No corr dimension found"));
        int ncorr = it->second;

        // //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, ncorr == CREATE_ANTENNA_JONES_NCORR,
            InvalidArgument("Number of correlations '",
                ncorr, "' does not equal '",
                CREATE_ANTENNA_JONES_NCORR, "'."));

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, ncorr});

        // Allocate the output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        using LTr = LaunchTraits<FT>;
        using Tr =  montblanc::kernel_traits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block = montblanc::shrink_small_dims(
            dim3(LTr::BLOCKDIMX, LTr::BLOCKDIMY, LTr::BLOCKDIMZ),
            ncorr*nchan, na, ntime);
        dim3 grid(montblanc::grid_from_thread_block(block,
            ncorr*nchan, na, ntime));

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the rime_create_antenna_jones CUDA kernel
        rime_create_antenna_jones<Tr> <<<grid, block, 0, device.stream()>>>(
            input_ptr<CT, typename Tr::CT>(bsqrt_list),
            input_ptr<CT, typename Tr::CT>(complex_phase_list),
            input_ptr<CT, typename Tr::CT>(feed_rotation_list),
            input_ptr<CT, typename Tr::CT>(ddes_list),
            reinterpret_cast<typename Tr::CT *>
                            (ant_jones_ptr->flat<CT>().data()),
            nsrc, ntime, na, nchan, ncorr);
    }

    template <typename TFType, typename GPUType>
    const GPUType *
    input_ptr(const tensorflow::OpInputList & in_list)
    {
        if(in_list.size() == 0)
            { return nullptr; }

        auto tensor_ptr = in_list[0].flat<TFType>().data();
        return reinterpret_cast<const GPUType *>(tensor_ptr);
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH

#endif // #if GOOGLE_CUDA
