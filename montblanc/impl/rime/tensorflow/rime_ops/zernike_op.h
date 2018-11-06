#ifndef ZERNIKE_DDE_ZERNIKE_OP_H
#define ZERNIKE_DDE_ZERNIKE_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_ZERNIKE_NAMESPACE_BEGIN namespace  {
#define MONTBLANC_ZERNIKE_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_ZERNIKE_NAMESPACE_BEGIN

// General definition of the Zernike op, which will be specialised in:
//   - zernike_op_cpu.h for CPUs
//   - zernike_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - zernike_op_cpu.cpp for CPUs
//   - zernike_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class Zernike {};

constexpr int _ZERNIKE_CORRS = 4;

MONTBLANC_ZERNIKE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef ZERNIKE_DDE_ZERNIKE_OP_H