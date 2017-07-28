#ifndef RIME_CIRCULAR_STOKES_SWAP_OP_H
#define RIME_CIRCULAR_STOKES_SWAP_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_BEGIN

// General definition of the CircularStokesSwap op, which will be specialised in:
//   - circular_stokes_swap_op_cpu.h for CPUs
//   - circular_stokes_swap_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - circular_stokes_swap_op_cpu.cpp for CPUs
//   - circular_stokes_swap_op_gpu.cu for CUDA devices
template <typename Device, typename FT>
class CircularStokesSwap {};

MONTBLANC_CIRCULAR_STOKES_SWAP_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CIRCULAR_STOKES_SWAP_OP_H