#include <string>
#include <vector>
#include <unordered_map>

#include "jones_multiply_op_cpu.h"
#include "shapes.h"

#include "tensorflow/core/framework/shape_inference.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;

using tensorflow::Status;


auto shape_function = [](InferenceContext* c)
{
    namespace tf = tensorflow;

    std::unordered_map<std::string, DimensionHandle> dim_sizes;

    std::vector<ShapeHandle> input_shapes;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->input("in", &input_shapes),
        "Unable to obtain input in");

    std::vector<std::string> str_schemas;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->GetAttr("schemas", &str_schemas),
        "Unable to obtain schemas");

    std::string str_output_schema;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        c->GetAttr("output_schema", &str_output_schema),
        "Unable to obtain output_schema");

    // Parse the output schema
    std::vector<std::string> output_schema;
    TF_RETURN_IF_ERROR(parse_shape_schema(str_output_schema, output_schema));

    if(input_shapes.size() != str_schemas.size())
    {
        return tf::errors::InvalidArgument("Number of inputs ",
            input_shapes.size(), " does not match the number of ",
            str_schemas.size());
    }

    // Figure out the dimension sizes from inputs and their
    // associated schemas
    for(int i=0; i < input_shapes.size(); ++i)
    {
        const ShapeHandle & shape = input_shapes[i];
        std::vector<std::string> schema;
        TF_RETURN_IF_ERROR(parse_shape_schema(str_schemas[i], schema));

        int ndims = c->Rank(shape);

        if(ndims != schema.size())
        {
            return tf::errors::InvalidArgument("Rank ", ndims,
                " of input ", i, " does not match the schema rank ",
                schema.size());
        }

        for(int d=0; d<ndims; ++d)
        {
            auto it = dim_sizes.find(schema[d]);

            if(it == dim_sizes.end())
                { dim_sizes.insert({schema[d], c->Dim(shape, d)}); }
            else
            {
                DimensionHandle tmp;

                TF_RETURN_WITH_CONTEXT_IF_ERROR(
                    c->Merge(c->Dim(shape, d), it->second, &tmp),
                    "Incompatible shapes");
            }
        }
    }

    // Create the final output schema
    std::vector<DimensionHandle> out_dims;

    for(auto & name: output_schema)
    {
        auto it = dim_sizes.find(name);
        out_dims.push_back(it == dim_sizes.end() ? c->MakeDim(1) : it->second);
    }

    c->set_output(0, c->MakeShape(out_dims));

    return Status::OK();
};

// Register the JonesMultiply operator.
REGISTER_OP("JonesMultiply")
    .Input("in: N * CT")
    .Output("out: CT")
    .Attr("N: int >= 1")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Attr("CT: {complex64, complex128} = DT_COMPLEX64")
    .Attr("schemas: list(string)")
    .Attr("output_schema: string = '(source,time,ant,chan,corr)'")
    .Doc(R"doc(Jones Matrix Multiplication)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for JonesMultiply
// handling permutation ['float', 'tensorflow::complex64']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<float>("FT")
    .TypeConstraint<tensorflow::complex64>("CT")
    .Device(tensorflow::DEVICE_CPU),
    JonesMultiply<CPUDevice, float, tensorflow::complex64>);

// Register a CPU kernel for JonesMultiply
// handling permutation ['double', 'tensorflow::complex128']
REGISTER_KERNEL_BUILDER(
    Name("JonesMultiply")
    .TypeConstraint<double>("FT")
    .TypeConstraint<tensorflow::complex128>("CT")
    .Device(tensorflow::DEVICE_CPU),
    JonesMultiply<CPUDevice, double, tensorflow::complex128>);



MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP
