#ifndef RIME_SUM_COHERENCIES_OP_H
#define RIME_SUM_COHERENCIES_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

// sum_coherencies namespace start and stop defines
#define MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN namespace sum_coherencies {
#define MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

// General definition of the SumCoherencies op, which will be specialised for CPUs and GPUs in
// sum_coherencies_op_cpu.h and sum_coherencies_op_gpu.cuh respectively, as well as float types (FT).
// Concrete template instantiations of this class should be provided in
// sum_coherencies_op_cpu.cpp and sum_coherencies_op_gpu.cu respectively
template <typename Device, typename FT, typename CT> class SumCoherencies {};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_H
