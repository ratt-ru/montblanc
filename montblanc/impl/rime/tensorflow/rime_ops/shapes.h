#ifndef _MONTBLANC_SHAPES_H_
#define _MONTBLANC_SHAPES_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

tensorflow::Status parse_shape_schema(const std::string & schema,
                                      std::vector<std::string> & result);

class TFOpKernel;
class TFShapeInference;

template <typename Context>
class TensorflowInputFacade;


template <>
class TensorflowInputFacade<TFOpKernel>
{
public:
    using DimSizes = std::unordered_map<std::string, int>;

private:
    tensorflow::OpKernelContext * context;
    std::unordered_map<std::string, DimSizes> input_dim_sizes;
    std::unordered_map<std::string, tensorflow::OpInputList> inputs;
    DimSizes input_dims;

    tensorflow::Status inspect_inputs(const std::string & name,
                                      const std::string & schema)
    {
        auto & input_list = inputs[name];
        TF_RETURN_IF_ERROR(context->input_list(name, &input_list));

        if(input_list.size() == 0)
            { return tensorflow::Status::OK(); }

        const tensorflow::Tensor & tensor = input_list[0];

        std::vector<std::string> schema_parts;
        TF_RETURN_IF_ERROR(parse_shape_schema(schema, schema_parts));

        if(schema_parts.size() != tensor.dims())
        {
            return tensorflow::errors::InvalidArgument(
                        "Number of shape schema parts (",
                        schema_parts.size(),
                        ") do not match input rank (",
                        tensor.dims(),
                        ") for input ", name);
        }

        // Dimension Sizes
        auto & dim_sizes = input_dim_sizes[name];

        // Assign
        for(std::size_t i = 0; i < schema_parts.size(); ++i)
            { dim_sizes.insert({schema_parts[i], tensor.dim_size(i)}); }

        return tensorflow::Status::OK();
    }


    tensorflow::Status merge()
    {
        namespace tf = tensorflow;

        for(const auto & ids: input_dim_sizes)
        {
            const auto & input_name = ids.first;
            const auto & dims = ids.second;

            for(const auto & d: dims)
            {
                const auto & dim_name = d.first;
                const auto & dim_value = d.second;

                // Is this dimension present in the output?
                auto it = input_dims.find(dim_name);

                // No, insert
                if(it == input_dims.end())
                {
                    input_dims.insert(d);
                }
                else if(dim_value != it->second)
                {
                    return tensorflow::errors::InvalidArgument(
                        "Input ", input_name,
                        " dimension ", dim_name,
                        " size ", dim_value,
                        " disagrees with new value ", it->second);
                }
            }
        }

        return tensorflow::Status::OK();
    }

public:
    TensorflowInputFacade(tensorflow::OpKernelContext * c)
         : context(c) {}


    tensorflow::Status inspect(
        std::vector<std::pair<std::string, std::string>> name_schemas)
    {
        for(const auto & name_schema : name_schemas)
        {
            TF_RETURN_IF_ERROR(inspect_inputs(std::get<0>(name_schema),
                                              std::get<1>(name_schema)));
        }

        TF_RETURN_IF_ERROR(merge());

        return tensorflow::Status::OK();
    }

    tensorflow::Status get_dim(const std::string & dim, int * size)
    {
        auto it = input_dims.find(dim);

        if(it == input_dims.end())
        {
            return tensorflow::errors::InvalidArgument("Dimension ",
                                                       dim, " not found.");
        }

        *size = it->second;
        return tensorflow::Status::OK();
    }

    tensorflow::Status get_tensor(const std::string & name,
                                  int index,
                                  const tensorflow::Tensor ** tensor)
    {
        auto it = inputs.find(name);

        if(it == inputs.end() || index >= it->second.size())
        {
            return tensorflow::errors::InvalidArgument("Input ",
                name, " at index ", index, " not found.");
        }

        *tensor = &it->second[index];
        return tensorflow::Status::OK();
    }
};


template <>
class TensorflowInputFacade<TFShapeInference>
{
private:
    using DimType = tensorflow::shape_inference::DimensionHandle;
    using DimSizes = std::unordered_map<std::string, DimType>;

private:
    tensorflow::shape_inference::InferenceContext * context;
    std::unordered_map<std::string, DimSizes> input_dim_sizes;
    std::unordered_map<std::string, tensorflow::OpInputList> inputs;
    DimSizes input_dims;

    tensorflow::Status inspect_inputs(const std::string & name)
    {
        using ShapeHandle = tensorflow::shape_inference::ShapeHandle;
        std::vector<ShapeHandle> input_vector;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(context->input(name, &input_vector),
            "Unable to obtain input " + name);

        // Argument not present, no checks
        if(input_vector.size() == 0)
            { return tensorflow::Status::OK(); }

        const ShapeHandle & shape = input_vector[0];

        // Attempt to obtain a schema
        std::string input_schema;
        tensorflow::Status status = context->GetAttr(name + "_schema",
                                                     &input_schema);

        // No schema, assume OK
        if(!status.ok())
            { return tensorflow::Status::OK(); }

        // Parse the shape schema
        std::vector<std::string> schema_parts;
        TF_RETURN_IF_ERROR(parse_shape_schema(input_schema, schema_parts));

        // Rank of schema should match rank of input shape
        if(schema_parts.size() != context->Rank(shape))
        {
            return tensorflow::errors::InvalidArgument(
                    "Number of shape schema parts (",
                    schema_parts.size(),
                    ") do not match input rank (",
                    context->Rank(shape),
                    ") for input ", name);
        }

        // Dimension Sizes
        auto & dim_sizes = input_dim_sizes[name];

        // Assign
        for(std::size_t i = 0; i < schema_parts.size(); ++i)
            { dim_sizes.insert({schema_parts[i], context->Dim(shape, i)}); }

        return tensorflow::Status::OK();
    }


    tensorflow::Status merge()
    {
        namespace tf = tensorflow;

        for(const auto & ids: input_dim_sizes)
        {
            const auto & input_name = ids.first;
            const auto & dims = ids.second;

            for(const auto & d: dims)
            {
                const auto & dim_name = d.first;
                const auto & dim_value = d.second;

                // Is this dimension present in the output?
                auto it = input_dims.find(dim_name);

                // No, insert
                if(it == input_dims.end())
                {
                    input_dims.insert(d);
                }
                else
                {
                    // Call tensorflow's dimension merge mechanism
                    // overwriting the existing value in input_dims
                    TF_RETURN_WITH_CONTEXT_IF_ERROR(
                        context->Merge(dim_value, it->second, &it->second),
                        "Couldn't merge dimension " + dim_name +
                        " from input " + input_name);
                }
            }
        }

        return tensorflow::Status::OK();
    }

public:
    TensorflowInputFacade(tensorflow::shape_inference::InferenceContext * c)
         : context(c) {}


    tensorflow::Status inspect(std::vector<std::string> names)
    {
        for(const auto & name : names)
        {
            TF_RETURN_IF_ERROR(inspect_inputs(name));
        }

        TF_RETURN_IF_ERROR(merge());

        return tensorflow::Status::OK();
    }

    tensorflow::Status get_dim(const std::string & dim, DimType * size)
    {
        auto it = input_dims.find(dim);

        if(it == input_dims.end())
        {
            return tensorflow::errors::InvalidArgument("Dimension ",
                                                       dim, " not found.");
        }

        *size = it->second;
        return tensorflow::Status::OK();
    }

    tensorflow::Status get_tensor(const std::string & name,
                                  int index,
                                  const tensorflow::Tensor ** tensor)
    {
        auto it = inputs.find(name);

        if(it == inputs.end() || index >= it->second.size())
        {
            return tensorflow::errors::InvalidArgument("Input ",
                name, " at index ", index, " not found.");
        }

        *tensor = &it->second[index];
        return tensorflow::Status::OK();
    }

};


#endif // #ifndef
