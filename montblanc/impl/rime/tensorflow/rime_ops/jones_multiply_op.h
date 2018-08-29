#ifndef RIME_JONES_MULTIPLY_OP_H
#define RIME_JONES_MULTIPLY_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

// General definition of the JonesMultiply op, which will be specialised in:
//   - jones_multiply_op_cpu.h for CPUs
//   - jones_multiply_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - jones_multiply_op_cpu.cpp for CPUs
//   - jones_multiply_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class JonesMultiply {};

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_JONES_MULTIPLY_OP_H