#ifndef RIME_JONES_MULTIPLY_OP_H
#define RIME_JONES_MULTIPLY_OP_H

#include <unordered_map>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

// montblanc namespace start and stop defines
#define MONTBLANC_NAMESPACE_BEGIN namespace montblanc {
#define MONTBLANC_NAMESPACE_STOP }

//  namespace start and stop defines
#define MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN namespace jones_multiply {
#define MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP }

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

constexpr int MAX_TENSOR_NDIM = 5;

// General definition of the JonesMultiply op, which will be specialised in:
//   - jones_multiply_op_cpu.h for CPUs
//   - jones_multiply_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - jones_multiply_op_cpu.cpp for CPUs
//   - jones_multiply_op_gpu.cu for CUDA devices
template <typename Device, typename FT, typename CT>
class JonesMultiply {};


tensorflow::Status infer_dimensionality(const tensorflow::OpInputList & in_list,
                                        const std::vector<std::string> & schemas,
                                        const std::string & str_output_schema,
                                        const std::vector<std::string> & output_schema,
                                        const std::unordered_map<std::string, int> & output_index,
                                        std::vector<std::vector<tensorflow::int64>> & reshapes,
                                        std::unordered_map<std::string, int> & output_sizes);

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_JONES_MULTIPLY_OP_H
