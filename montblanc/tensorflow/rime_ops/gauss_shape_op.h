#ifndef RIME_GAUSS_SHAPE_OP_H
#define RIME_GAUSS_SHAPE_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

// gauss_shape namespace start and stop defines
#define MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN namespace gauss_shape {
#define MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_GAUSS_SHAPE_NAMESPACE_BEGIN

// General definition of the GaussShape op, which will be specialised for CPUs and GPUs in
// gauss_shape_op_cpu.h and gauss_shape_op_gpu.cuh respectively.
// Concrete template instantiations of this class should be provided in
// gauss_shape_op_cpu.cpp and gauss_shape_op_gpu.cu respectively
template <typename Device, typename FT> class GaussShape {};

MONTBLANC_GAUSS_SHAPE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_GAUSS_SHAPE_OP_H
