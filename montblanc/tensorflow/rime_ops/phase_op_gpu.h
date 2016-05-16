#ifndef RIME_PHASE_OP_GPU_H_
#define RIME_PHASE_OP_GPU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename FT, typename CT>
struct RimePhaseGPU {
    static void compute(const GPUDevice & device,
        const FT * lm,
        const FT * uvw,
        const FT * frequency,
        const CT * complex_phase,
        int32 nsrc, int32 ntime, int32 na, int32 nchan);
};

}

#endif // #if GOOGLE_CUDA

#endif // #define RIME_PHASE_OP_GPU_H