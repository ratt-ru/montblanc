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
public:
    explicit CreateAntennaJones(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Sanity check the input tensors
        const tf::Tensor & in_bsqrt = context->input(0);
        const tf::Tensor & in_complex_phase = context->input(1);
        const tf::Tensor & in_feed_rotation = context->input(2);
        const tf::Tensor & in_ejones = context->input(3);

        // Extract problem dimensions
        int nsrc = in_complex_phase.dim_size(0);
        int ntime = in_complex_phase.dim_size(1);
        int na = in_complex_phase.dim_size(2);
        int nchan = in_complex_phase.dim_size(3);
        int npol = in_bsqrt.dim_size(3);

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
        auto ejones = in_ejones.tensor<CT, 5>();
        auto ant_jones = ant_jones_ptr->tensor<CT, 5>();

        #pragma omp parallel for collapse(3)
        for(int src=0; src < nsrc; ++src)
        {
            for(int time=0; time < ntime; ++time)
            {
                for(int ant=0; ant < na; ++ant)
                {
                    // Reference feed rotation matrix
                    const CT & l0 = feed_rotation(time, ant, 0);
                    const CT & l1 = feed_rotation(time, ant, 1);
                    const CT & l2 = feed_rotation(time, ant, 2);
                    const CT & l3 = feed_rotation(time, ant, 3);

                    for(int chan=0; chan < nchan; ++chan)
                    {
                        // Reference the complex phase
                        const CT & cp = complex_phase(src, time, ant, chan);

                        // Multiply complex phase by brightness square root
                        const CT kb0 = cp*bsqrt(src, time, chan, 0);
                        const CT kb1 = cp*bsqrt(src, time, chan, 1);
                        const CT kb2 = cp*bsqrt(src, time, chan, 2);
                        const CT kb3 = cp*bsqrt(src, time, chan, 3);

                        // Multiply in the feed rotation
                        const CT lkb0 = l0*kb0 + l1*kb2;
                        const CT lkb1 = l0*kb1 + l1*kb3;
                        const CT lkb2 = l2*kb0 + l3*kb2;
                        const CT lkb3 = l2*kb1 + l3*kb3;

                        // Reference ejones matrix
                        const CT & e0 = ejones(src, time, ant, chan, 0);
                        const CT & e1 = ejones(src, time, ant, chan, 1);
                        const CT & e2 = ejones(src, time, ant, chan, 2);
                        const CT & e3 = ejones(src, time, ant, chan, 3);

                        // Multiply in the dde term
                        ant_jones(src, time, ant, chan, 0) = e0*lkb0 + e1*lkb2;
                        ant_jones(src, time, ant, chan, 1) = e0*lkb1 + e1*lkb3;
                        ant_jones(src, time, ant, chan, 2) = e2*lkb0 + e3*lkb2;
                        ant_jones(src, time, ant, chan, 3) = e2*lkb1 + e3*lkb3;
                    }
                }
            }
        }
    }
};

MONTBLANC_CREATE_ANTENNA_JONES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_CPU_H
