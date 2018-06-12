#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"


using InferenceDimSizes = std::unordered_map<std::string, tensorflow::shape_inference::DimensionHandle>;
using InferenceInputDimSizes = std::unordered_map<std::string, InferenceDimSizes>;

using ComputeDimSizes = std::unordered_map<std::string, int>;
using ComputeInputDimSizes = std::unordered_map<std::string, ComputeDimSizes>;


tensorflow::Status get_input_and_schema_for_compute(
                         tensorflow::OpKernelContext * c,
                         const std::string & name,
                         const std::string & schema,
                         ComputeInputDimSizes & input_dim_sizes,
                         tensorflow::OpInputList & input_list);

tensorflow::Status get_input_and_schema_for_inference(
                         tensorflow::shape_inference::InferenceContext * c,
                         const std::string & name,
                         InferenceInputDimSizes & input_dim_sizes);

tensorflow::Status parse_shape_schema(const std::string & schema,
                        std::vector<std::string> & result);

tensorflow::Status merge_input_dims(
                        tensorflow::shape_inference::InferenceContext * c,
                        const InferenceInputDimSizes & input_dim_sizes,
                        InferenceDimSizes & input_dims);

tensorflow::Status merge_input_dims(
                        const ComputeInputDimSizes & input_dim_sizes,
                        ComputeDimSizes & input_dims);
