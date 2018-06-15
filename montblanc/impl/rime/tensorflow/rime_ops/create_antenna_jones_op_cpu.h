#ifndef RIME_CREATE_ANTENNA_JONES_OP_CPU_H
#define RIME_CREATE_ANTENNA_JONES_OP_CPU_H

#include "create_antenna_jones_op.h"
#include "shapes.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_BEGIN


// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;



// Specialise the CreateAntennaJones op for CPUs
template <typename FT, typename CT>
class CreateAntennaJones<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string bsqrt_schema;
    std::string complex_phase_schema;
    std::string feed_rotation_schema;
    std::string ddes_schema;
    tensorflow::Tensor dummy_CT_tensor;

public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        OP_REQUIRES_OK(context, context->GetAttr("bsqrt_schema", &bsqrt_schema));
        OP_REQUIRES_OK(context, context->GetAttr("complex_phase_schema", &complex_phase_schema));
        OP_REQUIRES_OK(context, context->GetAttr("feed_rotation_schema", &feed_rotation_schema));
        OP_REQUIRES_OK(context, context->GetAttr("ddes_schema", &ddes_schema));

        // Sanity check the output type vs the input types
        tf::DataType dtype;
        OP_REQUIRES_OK(context, context->GetAttr("CT", &dtype));

        std::vector<std::string> type_attrs = {"bsqrt_type",
                                                "complex_phase_type",
                                                "feed_rotation_type",
                                                "ddes_type"};

        for(const auto & type_attr: type_attrs)
        {
            tf::DataTypeVector dtypes;
            OP_REQUIRES_OK(context, context->GetAttr(type_attr, &dtypes));
            OP_REQUIRES(context, dtypes.size() <= 1,
                InvalidArgument(type_attr, " length > 1"));

            if(dtypes.size() == 1)
            {
                OP_REQUIRES(context, dtypes[0] == dtype,
                    InvalidArgument(type_attr, " ",
                        tf::DataTypeString(dtypes[0]),
                        " != output type ",
                        tf::DataTypeString(dtype)));
            }
        }

        // Create a dummy tensor representing non-existent inputs
        tf::TensorShape dummy_shape({1});

        OP_REQUIRES_OK(context, context->allocate_temp(
            tf::DataTypeToEnum<CT>::value,
            dummy_shape,
            &dummy_CT_tensor));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        ComputeInputDimSizes input_dim_sizes;

        tf::OpInputList bsqrt_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "bsqrt",
                                      bsqrt_schema,
                                      input_dim_sizes,
                                      bsqrt_list));

        tf::OpInputList complex_phase_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "complex_phase",
                                      complex_phase_schema,
                                      input_dim_sizes,
                                      complex_phase_list));

        tf::OpInputList feed_rotation_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "feed_rotation",
                                      feed_rotation_schema,
                                      input_dim_sizes,
                                      feed_rotation_list));

        tf::OpInputList ddes_list;
        OP_REQUIRES_OK(context, get_input_and_schema_for_compute(context,
                                      "ddes",
                                      ddes_schema,
                                      input_dim_sizes,
                                      ddes_list));

        ComputeDimSizes dim_sizes;
        OP_REQUIRES_OK(context, merge_input_dims(input_dim_sizes, dim_sizes));

        ComputeDimSizes::const_iterator it;
        ComputeDimSizes::const_iterator end = dim_sizes.end();

        OP_REQUIRES(context, (it = dim_sizes.find("source")) != end,
                    InvalidArgument("No source dimension found"));
        int nsrc = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("time")) != end,
                    InvalidArgument("No time dimension found"));
        int ntime = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("ant")) != end,
                    InvalidArgument("No ant dimension found"));
        int na = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("chan")) != end,
                    InvalidArgument("No chan dimension found"));
        int nchan = it->second;

        OP_REQUIRES(context, (it = dim_sizes.find("corr")) != end,
                    InvalidArgument("No corr dimension found"));
        int ncorr = it->second;

        // //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, ncorr == CREATE_ANTENNA_JONES_NCORR,
            InvalidArgument("Number of correlations '",
                ncorr, "' does not equal '",
                CREATE_ANTENNA_JONES_NCORR, "'."));

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, ncorr});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        bool have_bsqrt = bsqrt_list.size() > 0;
        bool have_complex_phase = complex_phase_list.size() > 0;
        bool have_feed_rotation = feed_rotation_list.size() > 0;
        bool have_ddes = ddes_list.size() > 0;

        const tf::Tensor & dummy_CT = dummy_CT_tensor;

        // Get flattened inputs
        auto bsqrt = have_bsqrt ?
                            bsqrt_list[0].flat<CT>() :
                            dummy_CT.flat<CT>();
        auto complex_phase = have_complex_phase ?
                            complex_phase_list[0].flat<CT>() :
                            dummy_CT.flat<CT>();
        auto feed_rotation = have_feed_rotation ?
                            feed_rotation_list[0].flat<CT>() :
                            dummy_CT.flat<CT>();
        auto ddes = have_ddes ?
                            ddes_list[0].flat<CT>() :
                            dummy_CT.flat<CT>();

        auto ant_jones = ant_jones_ptr->tensor<CT, 5>();

        #pragma omp parallel for collapse(3)
        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                for(int ant=0; ant < na; ++ant)
                {
                    for(int chan=0; chan < nchan; ++chan)
                    {
                        // Maintain a double buffer of complex matrix values
                        CT buf0[2];
                        CT buf1[2];
                        CT buf2[2];
                        CT buf3[2];
                        // active and inactive buffer indices
                        int a = 0;
                        int i = 1;
                        bool initialised = false;

                        if(have_bsqrt)
                        {
                            // Reference brightness square root
                            const int index = ((src*ntime + time)*nchan + chan)*ncorr;
                            const CT & b0 = bsqrt(index + 0);
                            const CT & b1 = bsqrt(index + 1);
                            const CT & b2 = bsqrt(index + 2);
                            const CT & b3 = bsqrt(index + 3);

                            if(initialised)
                            {
                                buf0[i] = b0*buf0[a] + b1*buf2[a];
                                buf1[i] = b0*buf1[a] + b1*buf3[a];
                                buf2[i] = b2*buf0[a] + b3*buf2[a];
                                buf3[i] = b2*buf1[a] + b3*buf3[a];
                            }
                            else
                            {
                                buf0[i] = b0;
                                buf1[i] = b1;
                                buf2[i] = b2;
                                buf3[i] = b3;
                                initialised = true;
                            }

                            std::swap(a, i);
                        }

                        if(have_complex_phase)
                        {
                            // Reference complex phase
                            int index = src*ntime + time;
                            index = (index*na + ant)*nchan + chan;
                            const CT & cp = complex_phase(index);

                            if(initialised)
                            {
                                buf0[i] = cp*buf0[a];
                                buf1[i] = cp*buf1[a];
                                buf2[i] = cp*buf2[a];
                                buf3[i] = cp*buf3[a];
                            }
                            else
                            {
                                buf0[i] = cp;
                                buf1[i] = cp;
                                buf2[i] = cp;
                                buf3[i] = cp;
                                initialised = true;
                            }

                            std::swap(a, i);
                        }

                        if(have_feed_rotation)
                        {
                            // Reference feed rotation matrix
                            const int index = (time*na + ant)*ncorr;
                            const CT & l0 = feed_rotation(index + 0);
                            const CT & l1 = feed_rotation(index + 1);
                            const CT & l2 = feed_rotation(index + 2);
                            const CT & l3 = feed_rotation(index + 3);

                            if(initialised)
                            {
                                buf0[i] = l0*buf0[a] + l1*buf2[a];
                                buf1[i] = l0*buf1[a] + l1*buf3[a];
                                buf2[i] = l2*buf0[a] + l3*buf2[a];
                                buf3[i] = l2*buf1[a] + l3*buf3[a];
                            }
                            else
                            {
                                buf0[i] = l0;
                                buf1[i] = l1;
                                buf2[i] = l2;
                                buf3[i] = l3;
                                initialised = true;
                            }

                            std::swap(a, i);
                        }


                        if(have_ddes)
                        {
                            // Reference ddes matrix
                            int index = ((src*ntime + time)*na + ant);
                            index = (index*nchan + chan)*ncorr;
                            const CT & e0 = ddes(index + 0);
                            const CT & e1 = ddes(index + 1);
                            const CT & e2 = ddes(index + 2);
                            const CT & e3 = ddes(index + 3);

                            if(initialised)
                            {
                                buf0[i] = e0*buf0[a] + e1*buf2[a];
                                buf1[i] = e0*buf1[a] + e1*buf3[a];
                                buf2[i] = e2*buf0[a] + e3*buf2[a];
                                buf3[i] = e2*buf1[a] + e3*buf3[a];
                            }
                            else
                            {
                                buf0[i] = e0;
                                buf1[i] = e1;
                                buf2[i] = e2;
                                buf3[i] = e3;
                                initialised = true;
                            }

                            std::swap(a, i);
                        }

                        // This shouldn't happen, use ID matrix
                        if(!initialised)
                        {
                            buf0[a] = { 1.0, 0.0 };
                            buf1[a] = { 0.0, 0.0 };
                            buf2[a] = { 0.0, 0.0 };
                            buf3[a] = { 1.0, 0.0 };
                        }

                        // Multiply in the dde term
                        ant_jones(src, time, ant, chan, 0) = buf0[a];
                        ant_jones(src, time, ant, chan, 1) = buf1[a];
                        ant_jones(src, time, ant, chan, 2) = buf2[a];
                        ant_jones(src, time, ant, chan, 3) = buf3[a];
                    }
                }
            }
        }
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_CPU_H
