#ifndef RIME_RADEC_TO_LM_OP_H
#define RIME_RADEC_TO_LM_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_RADEC_TO_LM_NAMESPACE_BEGIN

// General definition of the RadecToLm op, which will be specialised in:
//   - radec_to_lm_op_cpu.h for CPUs
//   - radec_to_lm_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - radec_to_lm_op_cpu.cpp for CPUs
//   - radec_to_lm_op_gpu.cu for CUDA devices
template <typename Device, typename FT>
class RadecToLm {};

MONTBLANC_RADEC_TO_LM_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_RADEC_TO_LM_OP_H
