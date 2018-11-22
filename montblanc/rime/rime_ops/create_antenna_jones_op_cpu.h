#ifndef RIME_CREATE_ANTENNA_JONES_OP_CPU_H
#define RIME_CREATE_ANTENNA_JONES_OP_CPU_H

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "create_antenna_jones_op.h"
#include "shapes.h"

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
    TensorflowInputFacade<TFOpKernel> in_facade;
public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context),
        in_facade({"bsqrt", "complex_phase", "feed_rotation", "ddes"})
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        OP_REQUIRES_OK(context, in_facade.inspect(context));

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

    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;
        using tensorflow::errors::InvalidArgument;

        typename TensorflowInputFacade<TFOpKernel>::OpInputData op_data;
        OP_REQUIRES_OK(context, in_facade.inspect(context, &op_data));

        int nsrc, ntime, na, nchan, ncorr;
        OP_REQUIRES_OK(context, op_data.get_dim("source", &nsrc));
        OP_REQUIRES_OK(context, op_data.get_dim("time", &ntime));
        OP_REQUIRES_OK(context, op_data.get_dim("ant", &na));
        OP_REQUIRES_OK(context, op_data.get_dim("chan", &nchan));
        OP_REQUIRES_OK(context, op_data.get_dim("corr", &ncorr));

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

        const tf::Tensor * bsqrt_ptr = nullptr;
        const tf::Tensor * complex_phase_ptr = nullptr;
        const tf::Tensor * feed_rotation_ptr = nullptr;
        const tf::Tensor * ddes_ptr = nullptr;

        bool have_bsqrt =
            op_data.get_tensor("bsqrt", 0, &bsqrt_ptr).ok();
        bool have_complex_phase =
            op_data.get_tensor("complex_phase", 0, &complex_phase_ptr).ok();
        bool have_feed_rotation =
            op_data.get_tensor("feed_rotation", 0, &feed_rotation_ptr).ok();
        bool have_ddes = op_data.get_tensor("ddes", 0, &ddes_ptr).ok();

        // Create a dummy tensor representing non-existent inputs
        const tf::Tensor dummy_CT(tf::DataTypeToEnum<CT>::value, {1});

        // Get flattened inputs
        auto bsqrt = have_bsqrt ? bsqrt_ptr->flat<CT>() :
                                dummy_CT.flat<CT>();
        auto complex_phase = have_complex_phase ?
                                complex_phase_ptr->flat<CT>() :
                                dummy_CT.flat<CT>();
        auto feed_rotation = have_feed_rotation ?
                                feed_rotation_ptr->flat<CT>() :
                                dummy_CT.flat<CT>();
        auto ddes = have_ddes ? ddes_ptr->flat<CT>() :
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
