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
        in_facade({"time_index", "antenna1", "antenna2",
                   "ant_scalar_1", "ant_jones_1",
                   "baseline_scalar", "baseline_jones",
                   "ant_scalar_2", "ant_jones_2",
                   "base_coherencies"})
    {
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx));
    }

    void Compute(tensorflow::OpKernelContext * ctx) override
    {
        namespace tf = tensorflow;

        typename TensorflowInputFacade<TFOpKernel>::OpInputData op_data;
        OP_REQUIRES_OK(ctx, in_facade.inspect(ctx, &op_data));

        int nvrow, nsrc, ntime, na, nchan, ncorr;
        OP_REQUIRES_OK(ctx, op_data.get_dim("row", &nvrow));
        OP_REQUIRES_OK(ctx, op_data.get_dim("source", &nsrc));
        OP_REQUIRES_OK(ctx, op_data.get_dim("time", &ntime));
        OP_REQUIRES_OK(ctx, op_data.get_dim("ant", &na));
        OP_REQUIRES_OK(ctx, op_data.get_dim("chan", &nchan));
        OP_REQUIRES_OK(ctx, op_data.get_dim("corr", &ncorr));

        int ncorrchan = nchan*ncorr;

        // Allocate an output tensor
        tf::Tensor * coherencies_ptr = nullptr;
        tf::TensorShape coherencies_shape = tf::TensorShape({
            nvrow, nchan, ncorr });
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, coherencies_shape, &coherencies_ptr));

        const tf::Tensor * time_index_ptr = nullptr;
        const tf::Tensor * antenna1_ptr = nullptr;
        const tf::Tensor * antenna2_ptr = nullptr;
        const tf::Tensor * ant_scalar_1_ptr = nullptr;
        const tf::Tensor * ant_jones_1_ptr = nullptr;
        const tf::Tensor * baseline_scalar_ptr = nullptr;
        const tf::Tensor * baseline_jones_ptr = nullptr;
        const tf::Tensor * ant_scalar_2_ptr = nullptr;
        const tf::Tensor * ant_jones_2_ptr = nullptr;
        const tf::Tensor * base_coherencies_ptr = nullptr;

        OP_REQUIRES_OK(ctx, op_data.get_tensor("time_index", 0,
                                                 &time_index_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna1", 0,
                                                 &antenna1_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna2", 0,
                                                 &antenna2_ptr));
        bool have_ant_1_scalar = op_data.get_tensor("ant_scalar_1", 0,
                                                 &ant_scalar_1_ptr).ok();
        OP_REQUIRES_OK(ctx, op_data.get_tensor("ant_jones_1", 0,
                                                 &ant_jones_1_ptr));
        bool have_bl_scalar = op_data.get_tensor("baseline_scalar", 0,
                                                 &baseline_scalar_ptr).ok();
        bool have_bl_jones = op_data.get_tensor("baseline_jones", 0,
                                                 &baseline_jones_ptr).ok();
        bool have_ant_2_scalar = op_data.get_tensor("ant_scalar_2", 0,
                                                 &ant_scalar_2_ptr).ok();
        OP_REQUIRES_OK(ctx, op_data.get_tensor("ant_jones_2", 0,
                                                 &ant_jones_2_ptr));
        bool have_base = op_data.get_tensor("base_coherencies", 0,
                                                 &base_coherencies_ptr).ok();

        // Dummy variables to handle the absence of inputs
        const tf::Tensor dummy_phase(tf::DataTypeToEnum<CT>::value, {1});
        const tf::Tensor dummy_base(tf::DataTypeToEnum<CT>::value, {1,1,1});
        const tf::Tensor dummy_ant_scalar(tf::DataTypeToEnum<CT>::value, {1,1,1,1,1});
        const tf::Tensor dummy_bl_scalar(tf::DataTypeToEnum<CT>::value, {1,1,1,1,});

        auto time_index = time_index_ptr->tensor<int,1>();
        auto antenna1 = antenna1_ptr->tensor<int,1>();
        auto antenna2 = antenna2_ptr->tensor<int,1>();
        auto ant_scalar_1 = have_ant_1_scalar ?
                        ant_scalar_1_ptr->tensor<CT, 5>() :
                        dummy_ant_scalar.tensor<CT, 5>();
        auto ant_jones_1 = ant_jones_1_ptr->tensor<CT, 5>();
        auto baseline_scalar = have_bl_scalar ?
                        baseline_scalar_ptr->tensor<CT, 4>() :
                        dummy_bl_scalar.tensor<CT, 4>();
        auto baseline_jones = have_bl_jones ?
                        baseline_jones_ptr->tensor<CT, 4>() :
                        dummy_bl_scalar.tensor<CT, 4>();
        auto ant_scalar_2 = have_ant_2_scalar ?
                        ant_scalar_2_ptr->tensor<CT, 5>() :
                        dummy_ant_scalar.tensor<CT, 5>();
        auto ant_jones_2 = ant_jones_2_ptr->tensor<CT, 5>();
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
                    // Reference antenna 1 jones
                    CT a0 = ant_jones_1(src, time, ant1, chan, 0);
                    CT a1 = ant_jones_1(src, time, ant1, chan, 1);
                    CT a2 = ant_jones_1(src, time, ant1, chan, 2);
                    CT a3 = ant_jones_1(src, time, ant1, chan, 3);

                    // Multiply in the scalar
                    if(have_ant_1_scalar)
                    {
                        a0 = ant_scalar_1(src, time, ant1, chan, 0) * a0;
                        a1 = ant_scalar_1(src, time, ant1, chan, 1) * a1;
                        a2 = ant_scalar_1(src, time, ant1, chan, 2) * a2;
                        a3 = ant_scalar_1(src, time, ant1, chan, 3) * a3;
                    }

                    // Handle the baseline scalar and jones
                    CT c0, c1, c2, c3;

                    if(have_bl_scalar && have_bl_jones)
                    {
                        CT b0 = baseline_jones(src, vrow, chan, 0);
                        CT b1 = baseline_jones(src, vrow, chan, 1);
                        CT b2 = baseline_jones(src, vrow, chan, 2);
                        CT b3 = baseline_jones(src, vrow, chan, 3);

                        b0 = baseline_scalar(src, vrow, chan, 0) * b0;
                        b1 = baseline_scalar(src, vrow, chan, 1) * b1;
                        b2 = baseline_scalar(src, vrow, chan, 2) * b2;
                        b3 = baseline_scalar(src, vrow, chan, 3) * b3;

                        // Multiply in antenna 1
                        c0 = a0*b0 + a1*b2;
                        c1 = a0*b1 + a1*b3;
                        c2 = a2*b0 + a3*b2;
                        c3 = a2*b1 + a3*b3;
                    }
                    else if(have_bl_scalar && !have_bl_jones)
                    {
                        CT b0 = baseline_scalar(src, vrow, chan, 0);
                        CT b1 = baseline_scalar(src, vrow, chan, 1);
                        CT b2 = baseline_scalar(src, vrow, chan, 2);
                        CT b3 = baseline_scalar(src, vrow, chan, 3);

                        // Multiply in antenna 1
                        c0 = a0*b0 + a1*b2;
                        c1 = a0*b1 + a1*b3;
                        c2 = a2*b0 + a3*b2;
                        c3 = a2*b1 + a3*b3;

                    }
                    else if(!have_bl_scalar && have_bl_jones)
                    {
                        CT b0 = baseline_jones(src, vrow, chan, 0);
                        CT b1 = baseline_jones(src, vrow, chan, 1);
                        CT b2 = baseline_jones(src, vrow, chan, 2);
                        CT b3 = baseline_jones(src, vrow, chan, 3);

                        /// Multiply in antenna 1
                        c0 = a0*b0 + a1*b2;
                        c1 = a0*b1 + a1*b3;
                        c2 = a2*b0 + a3*b2;
                        c3 = a2*b1 + a3*b3;
                    }
                    else
                    {
                        c0 = a0;
                        c1 = a1;
                        c2 = a2;
                        c3 = a3;
                    }

                    // transpose of antenna 2 jones
                    CT d0 = ant_jones_2(src, time, ant2, chan, 0);
                    CT d1 = ant_jones_2(src, time, ant2, chan, 2);
                    CT d2 = ant_jones_2(src, time, ant2, chan, 1);
                    CT d3 = ant_jones_2(src, time, ant2, chan, 3);

                    if(have_ant_2_scalar)
                    {
                        d0 = ant_scalar_2(src, time, ant2, chan, 0) * d0;
                        d1 = ant_scalar_2(src, time, ant2, chan, 2) * d1;
                        d2 = ant_scalar_2(src, time, ant2, chan, 1) * d2;
                        d3 = ant_scalar_2(src, time, ant2, chan, 3) * d3;
                    }

                    // Convert to conjugate transpose
                    d0 = std::conj(d0);
                    d1 = std::conj(d1);
                    d2 = std::conj(d2);
                    d3 = std::conj(d3);

                    // Multiply jones matrices and accumulate them
                    // in the sum terms
                    s0 += c0*d0 + c1*d2;
                    s1 += c0*d1 + c1*d3;
                    s2 += c2*d0 + c3*d2;
                    s3 += c2*d1 + c3*d3;
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
