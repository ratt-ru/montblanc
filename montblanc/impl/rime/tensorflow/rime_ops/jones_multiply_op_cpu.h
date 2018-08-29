#ifndef RIME_JONES_MULTIPLY_OP_CPU_H
#define RIME_JONES_MULTIPLY_OP_CPU_H

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS


#include "jones_multiply_op.h"
#include "shapes.h"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_JONES_MULTIPLY_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the JonesMultiply op for CPUs
template <typename FT, typename CT>
class JonesMultiply<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string str_output_schema;
    std::vector<std::string> schemas;
    std::vector<std::string> output_schema;
    std::unordered_map<std::string, int> output_index;
    int N;

public:
    explicit JonesMultiply(tensorflow::OpKernelConstruction * context)
        : tensorflow::OpKernel(context),
          str_output_schema("(source,time,ant,chan,corr)")
    {
        OP_REQUIRES_OK(context, context->GetAttr("schemas",
                                                 &schemas));
        OP_REQUIRES_OK(context, context->GetAttr("N", &N));


        OP_REQUIRES_OK(context,
            parse_shape_schema(str_output_schema, output_schema));

        for(int i=0; i < output_schema.size(); ++i)
            { output_index.insert({output_schema[i], i}); }
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        std::unordered_map<std::string, int> output_sizes;
        tensorflow::OpInputList in_list;
        context->input_list("in", & in_list);

        OP_REQUIRES(context, in_list.size() == schemas.size(),
            tf::errors::InvalidArgument("Number of schemas ", schemas.size(),
                                        " does not match number of inputs ",
                                        in_list.size()));

        std::vector<std::vector<tf::int64>> reshapes;
        reshapes.reserve(in_list.size());

        for(int i=0; i<in_list.size(); ++i)
        {
            // Get the tensor shape
            const tf::TensorShape shape = in_list[i].shape();

            // Get the associated shape schema
            std::vector<std::string> schema;
            OP_REQUIRES_OK(context, parse_shape_schema(schemas[i], schema));

            // Number of elements in shape and schema must match
            OP_REQUIRES(context, schema.size() == shape.dims(),
                tf::errors::InvalidArgument("schema ", schemas[i], " "
                                            "shape does not match "
                                            "in[", i, "].shape of ",
                                            shape.DebugString()));

            // Work out the dimension sizes needed to reshape
            // the tensor rank up to that of the output schema.
            // Introduce 1's for missing dimensions
            std::vector<tf::int64> reshape;
            reshape.reserve(output_schema.size());

            // Start out with all 1.
            for(int j=0; j<output_schema.size(); ++j)
                { reshape.push_back(1); }

            for(int j=0; j<schema.size(); ++j)
            {
                // Either set the output size for this
                // schema dimension or check that it matches
                // a previously found value
                auto size_it = output_sizes.find(schema[j]);

                if(size_it == output_sizes.end())
                {
                    output_sizes.insert({schema[j], shape.dim_size(j)});
                }
                else
                {
                    OP_REQUIRES(context,
                       size_it->second == shape.dim_size(j),
                       tf::errors::InvalidArgument("Existing size ",
                           size_it->second, " for dimension ", schema[j],
                           " does not match ", shape.dim_size(j),
                           " found in input tensor ", i));
                }


                // Find index of schema dimension in output schema
                auto it = output_index.find(schema[j]);

                OP_REQUIRES(context, it != output_index.end(),
                    tf::errors::InvalidArgument(schema[j], " is not part "
                                                "of the output schema ",
                                                str_output_schema));

                // Set the dimension size at the output index
                // to the shape size
                reshape[it->second] = shape.dim_size(j);
            }

            reshapes.emplace_back(reshape);
        }


        // Determine output tensor shape
        tf::TensorShape output_shape;

        for(int i=0; i<output_schema.size(); ++i)
        {
            auto it = output_sizes.find(output_schema[i]);

            // Set to 1 if we couldn't infer the size
            if(it == output_sizes.end())
                { output_shape.AddDim(1); }
            else
                { output_shape.AddDim(it->second); }
        }

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));


        auto out = output_ptr->tensor<CT, 5>();

        // Set the output tensor to identity
        #pragma omp parallel for collapse(4)
        for(int osrc=0; osrc < out.dimension(0); ++osrc)
        {
            for(int otime=0; otime < out.dimension(1); ++otime)
            {
                for(int oant=0; oant < out.dimension(2); ++oant)
                {
                    for(int ochan=0; ochan < out.dimension(3); ++ochan)
                    {
                        out(osrc, otime, oant, ochan, 0) = {1.0, 0.0};
                        out(osrc, otime, oant, ochan, 1) = {0.0, 0.0};
                        out(osrc, otime, oant, ochan, 2) = {0.0, 0.0};
                        out(osrc, otime, oant, ochan, 3) = {1.0, 0.0};
                    }
                }
            }
        }

        for(int i=0; i<in_list.size(); ++i)
        {
            const tf::Tensor & tensor = in_list[i];
            auto data = tensor.shaped<CT, 5>(reshapes[i]);

            int isrc_inc = data.dimension(0) == out.dimension(0) ? 1 : 0;
            int itime_inc = data.dimension(1) == out.dimension(1) ? 1 : 0;
            int iant_inc = data.dimension(2) == out.dimension(2) ? 1 : 0;
            int ichan_inc = data.dimension(3) == out.dimension(3) ? 1 : 0;

            for(int isrc=0, osrc=0; osrc < out.dimension(0);
                ++osrc, isrc += isrc_inc)
            {
                for(int itime=0, otime=0; otime < out.dimension(1);
                    ++otime, itime += itime_inc)
                {
                    for(int iant=0, oant=0; oant < out.dimension(2);
                        ++oant, iant += iant_inc)
                    {
                        for(int ichan=0, ochan=0; ochan < out.dimension(3);
                            ++ochan, ichan += ichan_inc)
                        {
                            const CT t0 = out(osrc, otime, oant, ochan, 0);
                            const CT t1 = out(osrc, otime, oant, ochan, 1);
                            const CT t2 = out(osrc, otime, oant, ochan, 2);
                            const CT t3 = out(osrc, otime, oant, ochan, 3);

                            const CT & i0 = data(isrc, itime, iant, ichan, 0);
                            const CT & i1 = data(isrc, itime, iant, ichan, 1);
                            const CT & i2 = data(isrc, itime, iant, ichan, 2);
                            const CT & i3 = data(isrc, itime, iant, ichan, 3);

                            out(osrc, otime, oant, ochan, 0) = t0*i0 + t1*i2;
                            out(osrc, otime, oant, ochan, 1) = t0*i1 + t1*i3;
                            out(osrc, otime, oant, ochan, 2) = t2*i0 + t3*i2;
                            out(osrc, otime, oant, ochan, 3) = t2*i1 + t3*i3;
                        }
                    }
                }
            }
        }
    }
};

MONTBLANC_JONES_MULTIPLY_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_JONES_MULTIPLY_OP_CPU_H
