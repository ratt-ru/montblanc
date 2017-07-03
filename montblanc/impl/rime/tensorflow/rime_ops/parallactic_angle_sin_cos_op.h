#ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_H
#define RIME_PARALLACTIC_ANGLE_SIN_COS_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_BEGIN

// General definition of the ParallacticAngleSinCos op, which will be specialised in:
//   - parallactic_angle_sin_cos_op_cpu.h for CPUs
//   - parallactic_angle_sin_cos_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - parallactic_angle_sin_cos_op_cpu.cpp for CPUs
//   - parallactic_angle_sin_cos_op_gpu.cu for CUDA devices
template <typename Device, typename FT>
class ParallacticAngleSinCos {};

MONTBLANC_PARALLACTIC_ANGLE_SIN_COS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_H