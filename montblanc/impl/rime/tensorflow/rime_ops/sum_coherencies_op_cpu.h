#ifndef RIME_SUM_COHERENCIES_OP_CPU_H
#define RIME_SUM_COHERENCIES_OP_CPU_H

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "sum_coherencies_op.h"
#include "shapes.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_SUM_COHERENCIES_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the SumCoherencies op for CPUs
template <typename FT, typename CT>
class SumCoherencies<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    TensorflowInputFacade<TFOpKernel> in_facade;

public:
    explicit SumCoherencies(tensorflow::OpKernelConstruction * ctx) :
        tensorflow::OpKernel(ctx),
        in_facade({"time_index",
                   "antenna1",
                   "antenna2",
                   "ant_jones_1",
                   "baseline_jones",
                   "ant_jones_2",
                   "base_coherencies"})
    {
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx));
    }

    void Compute(tensorflow::OpKernelContext * ctx) override
    {
        namespace tf = tensorflow;

        typename TensorflowInputFacade<TFOpKernel>::OpInputData op_data;
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx, &op_data));

        const tf::Tensor * time_index_ptr = nullptr;
        const tf::Tensor * antenna1_ptr = nullptr;
        const tf::Tensor * antenna2_ptr = nullptr;
        const tf::Tensor * ant_jones_1_ptr = nullptr;
        const tf::Tensor * baseline_jones_ptr = nullptr;
        const tf::Tensor * ant_jones_2_ptr = nullptr;
        const tf::Tensor * base_coherencies_ptr = nullptr;

        OP_REQUIRES_OK(ctx, op_data.get_tensor("time_index", 0,
                                                 &time_index_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna1", 0,
                                                 &antenna1_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna2", 0,
                                                 &antenna2_ptr));
        bool have_ant_1_jones = op_data.get_tensor("ant_jones_1", 0,
                                                 &ant_jones_1_ptr).ok();
        bool have_bl_jones = op_data.get_tensor("baseline_jones", 0,
                                                 &baseline_jones_ptr).ok();
        bool have_ant_2_jones = op_data.get_tensor("ant_jones_2", 0,
                                                 &ant_jones_2_ptr).ok();
        bool have_base = op_data.get_tensor("base_coherencies", 0,
                                                 &base_coherencies_ptr).ok();

        OP_REQUIRES(ctx, have_ant_1_jones || have_bl_jones || have_ant_2_jones,
            tf::errors::InvalidArgument("No Jones Terms were supplied"));

        int nvrow, nsrc, ntime = 0, na = 0, nchan, ncorr;
        OP_REQUIRES_OK(ctx, op_data.get_dim("row", &nvrow));
        OP_REQUIRES_OK(ctx, op_data.get_dim("source", &nsrc));
        // Without antenna jones terms, these may not be present
        op_data.get_dim("time", &ntime);
        op_data.get_dim("ant", &na);
        OP_REQUIRES_OK(ctx, op_data.get_dim("chan", &nchan));
        OP_REQUIRES_OK(ctx, op_data.get_dim("corr", &ncorr));

        int ncorrchan = nchan*ncorr;

        // Allocate an output tensor
        tf::Tensor * coherencies_ptr = nullptr;
        tf::TensorShape coherencies_shape = tf::TensorShape({
            nvrow, nchan, ncorr });
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, coherencies_shape, &coherencies_ptr));


        // Dummy variables to handle the absence of inputs
        const tf::Tensor dummy_base(tf::DataTypeToEnum<CT>::value, {1,1,1});
        const tf::Tensor dummy_ant_jones(tf::DataTypeToEnum<CT>::value, {1,1,1,1,1});
        const tf::Tensor dummy_bl_jones(tf::DataTypeToEnum<CT>::value, {1,1,1,1,});

        auto time_index = time_index_ptr->tensor<int,1>();
        auto antenna1 = antenna1_ptr->tensor<int,1>();
        auto antenna2 = antenna2_ptr->tensor<int,1>();
        auto ant_jones_1 = have_ant_1_jones ?
                        ant_jones_1_ptr->tensor<CT, 5>() :
                        dummy_ant_jones.tensor<CT, 5>();
        auto baseline_jones = have_bl_jones ?
                        baseline_jones_ptr->tensor<CT, 4>() :
                        dummy_bl_jones.tensor<CT, 4>();
        auto ant_jones_2 = have_ant_2_jones ?
                        ant_jones_2_ptr->tensor<CT, 5>() :
                        dummy_ant_jones.tensor<CT, 5>();
        auto base_coherencies = have_base ?
                        base_coherencies_ptr->tensor<CT, 3>() :
                        dummy_base.tensor<CT, 3>();
        auto coherencies = coherencies_ptr->tensor<CT, 3>();


        #pragma omp parallel for
        for(int vrow=0; vrow<nvrow; ++vrow)
        {
            // Antenna pairs for this baseline
            int ant1 = antenna1(vrow);
            int ant2 = antenna2(vrow);
            int time = time_index(vrow);

            for(int chan=0; chan<nchan; ++chan)
            {
                // Load in the input model visibilities
                CT s0 = have_base ? base_coherencies(vrow, chan, 0) : 0;
                CT s1 = have_base ? base_coherencies(vrow, chan, 1) : 0;
                CT s2 = have_base ? base_coherencies(vrow, chan, 2) : 0;
                CT s3 = have_base ? base_coherencies(vrow, chan, 3) : 0;

                for(int src=0; src<nsrc; ++src)
                {
                    CT r0 = { 1.0, 0.0 };
                    CT r1 = { 0.0, 0.0 };
                    CT r2 = { 0.0, 0.0 };
                    CT r3 = { 1.0, 0.0 };

                    if(have_ant_2_jones)
                    {
                        // conjugate transpose of antenna 2 jones
                        r0 = std::conj(ant_jones_2(src, time, ant2, chan, 0));
                        r1 = std::conj(ant_jones_2(src, time, ant2, chan, 2));
                        r2 = std::conj(ant_jones_2(src, time, ant2, chan, 1));
                        r3 = std::conj(ant_jones_2(src, time, ant2, chan, 3));
                    }

                    if(have_bl_jones)
                    {
                        const CT & b0 = baseline_jones(src, vrow, chan, 0);
                        const CT & b1 = baseline_jones(src, vrow, chan, 1);
                        const CT & b2 = baseline_jones(src, vrow, chan, 2);
                        const CT & b3 = baseline_jones(src, vrow, chan, 3);

                        CT t0 = b0*r0 + b1*r2;
                        CT t1 = b0*r1 + b1*r3;
                        CT t2 = b2*r0 + b3*r2;
                        CT t3 = b2*r1 + b3*r3;

                        r0 = t0;
                        r1 = t1;
                        r2 = t2;
                        r3 = t3;
                    }

                    // Initialise antenna 1 jones to identity
                    CT a0 = { 1.0, 0.0 };
                    CT a1 = { 0.0, 0.0 };
                    CT a2 = { 0.0, 0.0 };
                    CT a3 = { 1.0, 0.0 };

                    // Load antenna 1 if present
                    if(have_ant_1_jones)
                    {
                        a0 = ant_jones_1(src, time, ant1, chan, 0);
                        a1 = ant_jones_1(src, time, ant1, chan, 1);
                        a2 = ant_jones_1(src, time, ant1, chan, 2);
                        a3 = ant_jones_1(src, time, ant1, chan, 3);
                    }

                    // Multiply jones matrices and accumulate them
                    // in the sum terms
                    s0 += a0*r0 + a1*r2;
                    s1 += a0*r1 + a1*r3;
                    s2 += a2*r0 + a3*r2;
                    s3 += a2*r1 + a3*r3;
                }

                // Output accumulated model visibilities
                coherencies(vrow, chan, 0) = s0;
                coherencies(vrow, chan, 1) = s1;
                coherencies(vrow, chan, 2) = s2;
                coherencies(vrow, chan, 3) = s3;
            }
        }
    }
};

MONTBLANC_SUM_COHERENCIES_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_SUM_COHERENCIES_OP_CPU_H
