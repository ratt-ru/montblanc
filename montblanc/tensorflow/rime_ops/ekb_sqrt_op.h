#ifndef RIME_EKB_SQRT_OP_H
#define RIME_EKB_SQRT_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

// ekb_sqrt namespace start and stop defines
#define MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN namespace ekb_sqrt {
#define MONTBLANC_EKB_SQRT_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_EKB_SQRT_NAMESPACE_BEGIN

// General definition of the EKBSqrt op, which will be specialised for CPUs and GPUs in
// ekb_sqrt_op_cpu.h and ekb_sqrt_op_gpu.cuh respectively, as well as float types (FT).
// Concrete template instantiations of this class should be provided in
// ekb_sqrt_op_cpu.cpp and ekb_sqrt_op_gpu.cu respectively
template <typename Device, typename FT, typename CT> class EKBSqrt {};

MONTBLANC_EKB_SQRT_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_EKB_SQRT_OP_H
