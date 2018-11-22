#ifndef RIME_FEED_ROTATION_OP_H
#define RIME_FEED_ROTATION_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_FEED_ROTATION_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN

// General definition of the FeedRotation op, which will be specialised in:
//   - feed_rotation_op_cpu.h for CPUs
//   - feed_rotation_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - feed_rotation_op_cpu.cpp for CPUs
//   - feed_rotation_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class FeedRotation {};

// Number of polarisations handled by this kernel
constexpr int FEED_ROTATION_NPOL = 4;

MONTBLANC_FEED_ROTATION_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ROTATION_OP_H