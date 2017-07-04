#ifndef RIME_CREATE_ANTENNA_JONES_OP_H
#define RIME_CREATE_ANTENNA_JONES_OP_H

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

// create_antenna_jones namespace start and stop defines
#define MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN namespace create_antenna_jones {
#define MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN

// General definition of the CreateAntennaJones op, which will be specialised for CPUs and GPUs in
// create_antenna_jones_op_cpu.h and create_antenna_jones_op_gpu.cuh respectively, as well as float types (FT).
// Concrete template instantiations of this class should be provided in
// create_antenna_jones_op_cpu.cpp and create_antenna_jones_op_gpu.cu respectively
template <typename Device, typename FT, typename CT> class CreateAntennaJones {};

// Number of polarisations handled by this kernel
constexpr int CREATE_ANTENNA_JONES_NPOL = 4;

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_H
