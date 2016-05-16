#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device_base.h"

#include "phase_op_gpu.h"


namespace gpu = perftools::gputools;
//namespace gcudacc = platforms::gpus::gcudacc;


namespace tensorflow {

//gpu::Platform * platform = gpu::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

typedef Eigen::GpuDevice GPUDevice;

template <typename FT, typename CT>
class RimePhaseTraits;

template <>
class RimePhaseTraits<float, tensorflow::complex64>
{
public:
    typedef float FT;
    typedef float2 CT;
    typedef float2 lm_type;
    typedef float3 uvw_type;
    typedef float frequency_type;
    typedef float2 complex_phase_type;
};

template <>
class RimePhaseTraits<double, tensorflow::complex128>
{
public:
    typedef double FT;
    typedef double2 CT;
    typedef double2 lm_type;
    typedef double3 uvw_type;
    typedef double frequency_type;
    typedef double2 complex_phase_type;
};


template <typename Traits>
__global__ void rime_phase(
    typename Traits::lm_type * lm,
    typename Traits::uvw_type * uvw,
    typename Traits::frequency_type * frequency,
    typename Traits::complex_phase_type * complex_phase,
    int32 nsrc, int32 ntime, int32 na, int32 nchan)
{
    typedef typename Traits::FT FT;

    int chan = blockIdx.x*blockDim.x + threadIdx.x;
    int ant = blockIdx.y*blockDim.y + threadIdx.y;
    int time = blockIdx.z*blockDim.z + threadIdx.z;

    if(chan >= nchan || ant >= na || time >= ntime)
        { return; }

    for(int src=0; src < nsrc; ++src)
    {
        typename Traits::lm_type r_lm = lm[src];
        // TODO: This code doesn't do anything sensible
        FT n = FT(1.0) - r_lm[0]*r_lm[0] - r_lm[1]*r_lm[1];
        complex_phase[nsrc] = n;
    }
}

template <typename FT, typename CT>
void RimePhaseGPU<FT, CT>::compute(
        const GPUDevice & device,
        const FT * lm,
        const FT * uvw,
        const FT * frequency,
        const CT * complex_phase,
        int32 nsrc, int32 ntime, int32 na, int32 nchan)
{
    dim3 blocks;
    dim3 grid;

    typedef RimePhaseTraits<FT, CT> Traits;

    rime_phase<Traits> <<<grid, blocks, 0, device.stream()>>>(
        lm, uvw, frequency, complex_phase,
        nsrc, ntime, na, nchan);
}

//template<> struct RimePhaseGPU<float, tensorflow::complex64>;
//template<> struct RimePhaseGPU<double, tensorflow::complex128>;

} // namespace tensorflow

#endif // #if GOOGLE_CUDA