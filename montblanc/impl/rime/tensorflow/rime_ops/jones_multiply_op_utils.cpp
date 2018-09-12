#include "shapes.h"
#include "jones_multiply_op.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

tensorflow::Status infer_dimensionality(const tensorflow::OpInputList & in_list,
                                        const std::vector<std::string> & schemas,
                                        const std::string & str_output_schema,
                                        const std::vector<std::string> & output_schema,
                                        const std::unordered_map<std::string, int> & output_index,
                                        std::vector<std::vector<tensorflow::int64>> & reshapes,
                                        std::unordered_map<std::string, int> & output_sizes)
{
    namespace tf = tensorflow;

    if(output_schema.size() > MAX_TENSOR_NDIM)
    {
        return tf::errors::InvalidArgument("Output schema ",
                            output_schema.size(), " is greater than "
                            "the maximum number of tensor dimensions ",
                            MAX_TENSOR_NDIM);
    }


    for(int i=0; i<in_list.size(); ++i)
    {
        // Get the tensor shape
        const tf::TensorShape shape = in_list[i].shape();

        // Get the associated shape schema
        std::vector<std::string> schema;
        TF_RETURN_IF_ERROR(parse_shape_schema(schemas[i], schema));

        // Number of elements in shape and schema must match
        if(schema.size() != shape.dims())
        {
            return tf::errors::InvalidArgument("schema ", schemas[i], " "
                                               "shape does not match "
                                               "in[", i, "].shape of ",
                                               shape.DebugString());
        }

        // Work out the dimension sizes needed to reshape
        // the tensor rank up to that of the output schema.
        // Introduce 1's for missing dimensions
        std::vector<tf::int64> reshape;
        reshape.reserve(MAX_TENSOR_NDIM);

        for(int j=0; j<MAX_TENSOR_NDIM; ++j)
            { reshape.push_back(1); }

        for(int j=0; j<schema.size(); ++j)
        {
            // Either set the output size for this
            // schema dimension or check that it matches
            // a previously discovered value
            auto size_it = output_sizes.find(schema[j]);

            if(size_it == output_sizes.end())
            {
                output_sizes.insert({schema[j], shape.dim_size(j)});
            }
            else if(size_it->second != shape.dim_size(j))
            {
                return tf::errors::InvalidArgument("Existing size ",
                            size_it->second, " for dimension ", schema[j],
                            " does not match ", shape.dim_size(j),
                            " found in input tensor ", i);
            }


            // Find index of schema dimension in output schema
            auto it = output_index.find(schema[j]);

            if(it == output_index.end())
            {
                return tf::errors::InvalidArgument(schema[j], " is not part "
                                                   "of the output schema ",
                                                   str_output_schema);
            }

            // Set the dimension size at the output index
            // to the shape size
            reshape[it->second] = shape.dim_size(j);
        }

        reshapes.emplace_back(std::move(reshape));
    }

    return tf::Status::OK();
}

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
