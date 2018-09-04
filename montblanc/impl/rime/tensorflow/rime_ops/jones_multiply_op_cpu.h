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
    bool squeeze;
    int N;

public:
    explicit JonesMultiply(tensorflow::OpKernelConstruction * context)
        : tensorflow::OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("schemas", &schemas));
        OP_REQUIRES_OK(context, context->GetAttr("N", &N));
        OP_REQUIRES_OK(context, context->GetAttr("squeeze", &squeeze));
        OP_REQUIRES_OK(context, context->GetAttr("output_schema", &str_output_schema));


        OP_REQUIRES_OK(context,
            parse_shape_schema(str_output_schema, output_schema));

        for(int i=0; i < output_schema.size(); ++i)
            { output_index.insert({output_schema[i], i}); }
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        tensorflow::OpInputList in_list;
        context->input_list("jones_in", & in_list);

        OP_REQUIRES(context, in_list.size() == schemas.size(),
            tf::errors::InvalidArgument("Number of schemas ", schemas.size(),
                                        " does not match number of inputs ",
                                        in_list.size()));

        std::vector<std::vector<tf::int64>> reshapes;
        reshapes.reserve(in_list.size());
        std::unordered_map<std::string, int> output_sizes;

        OP_REQUIRES_OK(context, infer_dimensionality(in_list,
                                 schemas, str_output_schema,
                                 output_schema, output_index, reshapes,
                                 output_sizes));

        // Determine output tensor shape, this may be < MAX_TENSOR_NDIM
        tf::TensorShape output_shape;
        // Reshape output tensor to MAX_TENSOR_NDIM
        std::vector<tf::int64> out_reshape;

        OP_REQUIRES(context, MAX_TENSOR_NDIM - output_schema.size() >= 0,
                tf::errors::InvalidArgument("Output schema size ",
                    output_schema.size(), " exceeds ", MAX_TENSOR_NDIM));

        for(int i=0; i < MAX_TENSOR_NDIM - output_schema.size(); ++i)
        {
            out_reshape.push_back(1);
            //output_shape.AddDim(1);
        }

        for(int i=0; i < output_schema.size(); ++i)
        {
            // Was this output dimension in the inputs?
            auto it = output_sizes.find(output_schema[i]);

            // No
            if(it == output_sizes.end())
            {
                out_reshape.push_back(1);

                // Ignore if we're squeezing else set to 1
                if(squeeze)
                    { continue; }

                output_shape.AddDim(1);
            }
            else
            {
                out_reshape.push_back(it->second);
                output_shape.AddDim(it->second);
            }
        }

        // Allocate an output tensor
        tf::Tensor * output_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, output_shape, &output_ptr));

        OP_REQUIRES(context, out_reshape.size() == MAX_TENSOR_NDIM,
            tf::errors::InvalidArgument("Mismatch"));

        auto out = output_ptr->shaped<CT, MAX_TENSOR_NDIM>(out_reshape);

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
            auto data = tensor.shaped<CT, MAX_TENSOR_NDIM>(reshapes[i]);

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
