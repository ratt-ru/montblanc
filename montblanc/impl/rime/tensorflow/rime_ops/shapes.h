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
    using InputDimSizes = std::unordered_map<std::string, DimSizes>;

    using SchemaParts = std::vector<std::string>;
    using InputSchemas = std::unordered_map<std::string, SchemaParts>;

    using InputLists = std::unordered_map<std::string, tensorflow::OpInputList>;

    class OpInputData
    {
    public:
        DimSizes dim_sizes;
        InputLists input_lists;

        OpInputData() = default;

        tensorflow::Status merge(const InputDimSizes & input_dim_sizes)
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
                    auto it = dim_sizes.find(dim_name);

                    // No, insert
                    if(it == dim_sizes.end())
                    {
                        dim_sizes.insert(d);
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


        tensorflow::Status construct(tensorflow::OpKernelContext * ctx,
                                    const std::vector<std::string> & input_names,
                                    const InputSchemas & schemas)
        {
            InputDimSizes input_dim_sizes;

            for(const auto & input_name : input_names)
            {
                auto & input_list = input_lists[input_name];
                TF_RETURN_IF_ERROR(ctx->input_list(input_name, &input_list));

                // An empty list is valid
                if(input_list.size() == 0)
                    { continue; }

                const tensorflow::Tensor & tensor = input_list[0];

                auto it = schemas.find(input_name);

                // No schema exists for this input, so we can't
                // deduce symbolic dimensions
                if(it == schemas.end())
                    { continue; }

                auto & schema_parts = it->second;

                if(schema_parts.size() != tensor.dims())
                {
                    return tensorflow::errors::InvalidArgument(
                                "Number of shape schema parts (",
                                schema_parts.size(),
                                ") do not match input rank (",
                                tensor.dims(),
                                ") for input ", input_name);
                }

                // Dimension Sizes
                auto & dim_sizes = input_dim_sizes[input_name];

                // Assign
                for(std::size_t i = 0; i < schema_parts.size(); ++i)
                    { dim_sizes.insert({schema_parts[i], tensor.dim_size(i)}); }
            }

            TF_RETURN_IF_ERROR(merge(input_dim_sizes));
            return tensorflow::Status::OK();
        }

        tensorflow::Status get_dim(const std::string & dim, int * size)
        {
            auto it = dim_sizes.find(dim);

            if(it == dim_sizes.end())
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
            auto it = input_lists.find(name);

            if(it == input_lists.end() || index >= it->second.size())
            {
                return tensorflow::errors::InvalidArgument("Input ",
                    name, " at index ", index, " not found.");
            }

            *tensor = &it->second[index];
            return tensorflow::Status::OK();
        }

    };



private:
    std::vector<std::string> input_names;
    InputSchemas schemas;
    std::unordered_map<std::string, DimSizes> input_dim_sizes;
    std::unordered_map<std::string, tensorflow::OpInputList> inputs;
    DimSizes input_dims;

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
    TensorflowInputFacade(const std::vector<std::string> & input_names_)
        : input_names(input_names_) {}

    tensorflow::Status inspect(tensorflow::OpKernelConstruction * ctx)
    {
        for(const auto & input_name : input_names)
        {
            std::string schema_name = input_name + "_schema";
            std::string schema;

            tensorflow::Status status = ctx->GetAttr(schema_name, &schema);

            if(!status.ok())
                {  continue;  }

            auto it = schemas.find(schema);

            if(it != schemas.end())
            {
                return tensorflow::errors::InvalidArgument(
                    "Schema for input ", input_name, " already exists ");
            }

            std::vector<std::string> schema_parts;
            TF_RETURN_IF_ERROR(parse_shape_schema(schema, schema_parts));

            schemas.insert({input_name, std::move(schema_parts)});
        }

        return tensorflow::Status::OK();
    }

    tensorflow::Status inspect(tensorflow::OpKernelContext * ctx,
                            OpInputData * op_input_data)
    {
        TF_RETURN_IF_ERROR(op_input_data->construct(ctx, input_names, schemas));
        return tensorflow::Status::OK();
    }
};


template <>
class TensorflowInputFacade<TFShapeInference>
{
private:
    using DimType = tensorflow::shape_inference::DimensionHandle;
    using DimSizes = std::unordered_map<std::string, DimType>;
    using ShapeHandle = tensorflow::shape_inference::ShapeHandle;
    using ShapeHandles = std::vector<ShapeHandle>;

private:
    tensorflow::shape_inference::InferenceContext * context;
    std::unordered_map<std::string, DimSizes> input_dim_sizes;
    std::unordered_map<std::string, std::vector<ShapeHandle>> inputs;
    DimSizes input_dims;

    tensorflow::Status inspect_inputs(const std::string & name)
    {
        auto input_vector = inputs[name];
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
            { TF_RETURN_IF_ERROR(inspect_inputs(name)); }

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
                                  const ShapeHandle ** shape_handle)
    {
        auto it = inputs.find(name);

        if(it == inputs.end() || index >= it->second.size())
        {
            return tensorflow::errors::InvalidArgument("Input ",
                name, " at index ", index, " not found.");
        }

        *shape_handle = &it->second[index];
        return tensorflow::Status::OK();
    }
};


#endif // #ifndef
