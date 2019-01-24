#ifndef RIME_FEED_ANGLE_OP_H
#define RIME_FEED_ANGLE_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN namespace feed_angle {
#define MONTBLANC_FEED_ANGLE_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN

// General definition of the FeedAngle op, which will be specialised in:
//   - feed_angle_op_cpu.h for CPUs
//   - feed_angle_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - feed_angle_op_cpu.cpp for CPUs
//   - feed_angle_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class FeedAngle {};

MONTBLANC_FEED_ANGLE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ANGLE_OP_H
