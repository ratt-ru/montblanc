#ifndef RIME_CREATE_ANTENNA_JONES_OP_CPU_H
#define RIME_CREATE_ANTENNA_JONES_OP_CPU_H

#include "create_antenna_jones_op.h"

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
    bool have_bsqrt;
    bool have_complex_phase;
    bool have_feed_rotation;
    bool have_ddes;

public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context),
        have_bsqrt(false),
        have_complex_phase(false),
        have_feed_rotation(false),
        have_ddes(false)
    {
        OP_REQUIRES_OK(context, context->GetAttr("have_bsqrt",
                                                 &have_bsqrt));
        OP_REQUIRES_OK(context, context->GetAttr("have_complex_phase",
                                                 &have_complex_phase));
        OP_REQUIRES_OK(context, context->GetAttr("have_feed_rotation",
                                                 &have_feed_rotation));
        OP_REQUIRES_OK(context, context->GetAttr("have_ddes",
                                                 &have_ddes));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_bsqrt = context->input(0);
        const tf::Tensor & in_complex_phase = context->input(1);
        const tf::Tensor & in_feed_rotation = context->input(2);
        const tf::Tensor & in_ddes = context->input(3);

        int nsrc = -1, ntime = -1, na = -1, nchan = -1, npol = -1;

        auto update_dim = [](int & old_size,
                            const tf::Tensor & tensor,
                            int dim) -> tf::Status
        {
            auto new_size = tensor.dim_size(dim);

            if(old_size == -1)
            {
                old_size = new_size;
            }
            else if(old_size != new_size)
            {
                return tf::Status(tf::errors::InvalidArgument(
                        "Previously set dimension size '",  old_size,
                        "' does not equal new size '", new_size, "'"));
            }

            return tf::Status::OK();
        };

        if(have_bsqrt)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_bsqrt, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_bsqrt, 1));
            OP_REQUIRES_OK(context, update_dim(nchan, in_bsqrt, 2));
            OP_REQUIRES_OK(context, update_dim(npol, in_bsqrt, 3));
        }

        if(have_complex_phase)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_complex_phase, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_complex_phase, 1));
            OP_REQUIRES_OK(context, update_dim(na, in_complex_phase, 2));
            OP_REQUIRES_OK(context, update_dim(nchan, in_complex_phase, 3));
        }

        if(have_feed_rotation)
        {
            OP_REQUIRES_OK(context, update_dim(ntime, in_feed_rotation, 0));
            OP_REQUIRES_OK(context, update_dim(na, in_feed_rotation, 1));
        }

        if(have_ddes)
        {
            OP_REQUIRES_OK(context, update_dim(nsrc, in_ddes, 0));
            OP_REQUIRES_OK(context, update_dim(ntime, in_ddes, 1));
            OP_REQUIRES_OK(context, update_dim(na, in_ddes, 2));
            OP_REQUIRES_OK(context, update_dim(nchan, in_ddes, 3));
            OP_REQUIRES_OK(context, update_dim(npol, in_ddes, 4));
        }

        //GPU kernel above requires this hard-coded number
        OP_REQUIRES(context, npol == CREATE_ANTENNA_JONES_NPOL,
            tf::errors::InvalidArgument("Number of polarisations '",
                npol, "' does not equal '", CREATE_ANTENNA_JONES_NPOL, "'."));

        tf::TensorShape ant_jones_shape({nsrc, ntime, na, nchan, npol});

        // Allocate an output tensor
        tf::Tensor * ant_jones_ptr = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, ant_jones_shape, &ant_jones_ptr));

        // Get pointers to flattened tensor data buffers
        auto bsqrt = in_bsqrt.tensor<CT, 4>();
        auto complex_phase = in_complex_phase.tensor<CT, 4>();
        auto feed_rotation = in_feed_rotation.tensor<CT, 3>();
        auto ddes = in_ddes.tensor<CT, 5>();
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
                            const int index = ((src*ntime + time)*nchan + chan)*npol;
                            const CT & b0 = bsqrt(src, time, chan, 0);
                            const CT & b1 = bsqrt(src, time, chan, 1);
                            const CT & b2 = bsqrt(src, time, chan, 2);
                            const CT & b3 = bsqrt(src, time, chan, 3);

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
                            const CT & cp = complex_phase(src, time, ant, chan);

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
                            const CT & l0 = feed_rotation(time, ant, 0);
                            const CT & l1 = feed_rotation(time, ant, 1);
                            const CT & l2 = feed_rotation(time, ant, 2);
                            const CT & l3 = feed_rotation(time, ant, 3);

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
                            const CT & e0 = ddes(src, time, ant, chan, 0);
                            const CT & e1 = ddes(src, time, ant, chan, 1);
                            const CT & e2 = ddes(src, time, ant, chan, 2);
                            const CT & e3 = ddes(src, time, ant, chan, 3);

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
