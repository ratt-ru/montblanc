#ifndef RIME_BRIGHTNESS_OP_H
#define RIME_BRIGHTNESS_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_BRIGHTNESS_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_BRIGHTNESS_NAMESPACE_BEGIN

// General definition of the Brightness op, which will be specialised in:
//   - brightness_op_cpu.h for CPUs
//   - brightness_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - brightness_op_cpu.cpp for CPUs
//   - brightness_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class Brightness {};

MONTBLANC_BRIGHTNESS_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_BRIGHTNESS_OP_H