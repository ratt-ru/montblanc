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

