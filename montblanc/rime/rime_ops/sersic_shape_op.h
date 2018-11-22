#ifndef RIME_SERSIC_SHAPE_OP_H
#define RIME_SERSIC_SHAPE_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

// sersic_shape namespace start and stop defines
#define MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN namespace sersic_shape {
#define MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SERSIC_SHAPE_NAMESPACE_BEGIN

// General definition of the SersicShape op, which will be specialised for CPUs and GPUs in
// sersic_shape_op_cpu.h and sersic_shape_op_gpu.cuh respectively.
// Concrete template instantiations of this class should be provided in
// sersic_shape_op_cpu.cpp and sersic_shape_op_gpu.cu respectively
template <typename Device, typename FT> class SersicShape {};

MONTBLANC_SERSIC_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SERSIC_SHAPE_OP_H
