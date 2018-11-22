#ifndef RIME_POST_PROCESS_VISIBILITIES_OP_H
#define RIME_POST_PROCESS_VISIBILITIES_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_BEGIN

// General definition of the PostProcessVisibilities op, which will be specialised in:
//   - post_process_visibilities_op_cpu.h for CPUs
//   - post_process_visibilities_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - post_process_visibilities_op_cpu.cpp for CPUs
//   - post_process_visibilities_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class PostProcessVisibilities {};

MONTBLANC_POST_PROCESS_VISIBILITIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_POST_PROCESS_VISIBILITIES_OP_H