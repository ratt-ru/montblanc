#include "shapes.h"

// Parses shape schema string "(source,ant,(x,y,z))"
// into a std::vector<std::string> = { "source", "ant", "(x,y,z)"}
tensorflow::Status parse_shape_schema(const std::string & schema,
                                      std::vector<std::string> & result)
{
    namespace tf = tensorflow;

    // Insist on parentheses
    if(schema[0] != '(' || schema[schema.size()-1] != ')')
    {
        return tf::errors::InvalidArgument("Shape schema \"", schema,
                                           "\" is not surrounded "
                                           "by parentheses (...)");
    }

    // Mark the '(' token as a split index
    std::vector<int> indices = { 0 };
    std::size_t depth = 1;

    for(std::size_t i = 1; i < schema.size()-1; ++i)
    {
        if(schema[i] == '(')
            { depth += 1; }
        else if(schema[i] == ')')
            { depth -= 1; }
        // If we're still between the first '(' and ')'
        // tokens, mark commas as an index to split on
        else if(depth == 1 && schema[i] == ',')
            { indices.push_back(i); }
    }

    // Mark the ')' token as a split index
    indices.push_back(schema.size()-1);

    // Extract dimensions from individual ranges
    for(std::size_t i = 0; i < indices.size() - 1; ++i)
    {
        // Identify start and end of the range
        auto start_i = indices[i]+1;   // +1 -- don't include split token
        auto end_i = indices[i+1];

        // Trim whitespace
        for(; start_i < end_i && std::isspace(schema[start_i]); ++start_i) {}
        for(; start_i < end_i && std::isspace(schema[end_i-1]); --end_i) {}

        // Ignore empty ranges
        if(start_i == end_i)
            { continue; }

        // Add dimension between the start and ending iterators
        auto start_it = schema.begin() + start_i;
        auto end_it = schema.begin() + end_i;
        result.emplace_back(std::string(start_it, end_it));
    }

    return tf::Status::OK();
}



tensorflow::Status get_input_and_schema_for_compute(
                        tensorflow::OpKernelContext * c,
                        const std::string & name,
                        const std::string & schema,
                        ComputeInputDimSizes & input_dim_sizes,
                        tensorflow::OpInputList & input_list)
{
    namespace tf = tensorflow;
    using tensorflow::errors::InvalidArgument;

    TF_RETURN_IF_ERROR(c->input_list(name, &input_list));

    // Argument not present, no checks
    if(input_list.size() == 0)
        { return tf::Status::OK(); }

    if(input_list.size() > 1)
    {
        return InvalidArgument("More than one input received "
                               "for input " + name);
    }

    const tf::Tensor & tensor = input_list[0];

    std::vector<std::string> schema_parts;
    TF_RETURN_IF_ERROR(parse_shape_schema(schema, schema_parts));

    // Rank of schema should match rank of input shape
    if(schema_parts.size() != tensor.dims())
    {
        return InvalidArgument("Number of shape schema parts (",
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

    return tf::Status::OK();
}


tensorflow::Status get_input_and_schema_for_inference(
                     tensorflow::shape_inference::InferenceContext * c,
                     const std::string & name,
                     InferenceInputDimSizes & input_dim_sizes)
{
    namespace tf = tensorflow;
    using tensorflow::errors::InvalidArgument;
    using tensorflow::shape_inference::ShapeHandle;

    tf::Status status;
    std::vector<ShapeHandle> input_vector;
    std::string input_schema;

    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->input(name, &input_vector),
        "Unable to obtain input " + name);

    // Argument not present, no checks
    if(input_vector.size() == 0)
        { return tf::Status::OK(); }

    if(input_vector.size() > 1)
    {
        return InvalidArgument("More than one input received for "
                               "input " + name);
    }

    const ShapeHandle & shape = input_vector[0];

    // Attempt to obtain a schema
    status = c->GetAttr(name + "_schema", &input_schema);

    // No schema, assume OK
    if(!status.ok())
        { return tf::Status::OK(); }

    // Parse the shape schema
    std::vector<std::string> schema_parts;
    TF_RETURN_IF_ERROR(parse_shape_schema(input_schema, schema_parts));

    // Rank of schema should match rank of input shape
    if(schema_parts.size() != c->Rank(shape))
    {
        return InvalidArgument("Number of shape schema parts (",
                               schema_parts.size(),
                               ") do not match input rank (",
                               c->Rank(shape),
                               ") for input ", name);
    }

    // Dimension Sizes
    auto & dim_sizes = input_dim_sizes[name];

    // Assign
    for(std::size_t i = 0; i < schema_parts.size(); ++i)
        { dim_sizes.insert({schema_parts[i], c->Dim(shape, i)}); }

    return tf::Status::OK();
}




tensorflow::Status merge_input_dims(
                        tensorflow::shape_inference::InferenceContext * c,
                        const InferenceInputDimSizes & input_dim_sizes,
                        InferenceDimSizes & input_dims)
{
    namespace tf = tensorflow;
    using tensorflow::errors::InvalidArgument;

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
                    c->Merge(dim_value, it->second, &it->second),
                    "Couldn't merge dimension " + dim_name +
                    " from input " + input_name);
            }
        }
    }

    return tensorflow::Status::OK();
}




tensorflow::Status merge_input_dims(const ComputeInputDimSizes & input_dim_sizes,
                                    ComputeDimSizes & input_dims)
{
    namespace tf = tensorflow;
    using tensorflow::errors::InvalidArgument;

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
                return InvalidArgument("Input ", input_name,
                           " dimension ", dim_name, " size ", dim_value,
                           " disagrees with new value ", it->second);
            }
        }
    }

    return tensorflow::Status::OK();
}



// #include <iostream>
// int main(void)
// {
//     std::vector<std::string> cases = {
//         "(source,time,ant,(x,y,z))",
//         "(source,ant,chan)",
//         "(source,)",
//         "(source)",
//         "(bpadf"
//     };


//     for(const auto & schema: cases) {
//         std::vector<std::string> result;
//         tensorflow::Status status = parse_shape_schema(schema, result);

//         if(!status.ok())
//             { std::cout << status << std::endl; }
//         else
//         {
//             std::cout << "Dimensions: ";
//             for(const auto & dim: result)
//             {
//                 std::cout << dim << ",";
//             }
//             std::cout << std::endl;
//         }
//     }

// }

