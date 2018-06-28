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
        in_facade({"time_index", "antenna1", "antenna2", "shape",
                   "ant_jones", "sgn_brightness", "complex_phase",
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
        const tf::Tensor * shape_ptr = nullptr;
        const tf::Tensor * ant_jones_ptr = nullptr;
        const tf::Tensor * complex_phase_ptr = nullptr;
        const tf::Tensor * sgn_brightness_ptr = nullptr;
        const tf::Tensor * base_coherencies_ptr = nullptr;

        OP_REQUIRES_OK(ctx, op_data.get_tensor("time_index", 0,
                                                 &time_index_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna1", 0,
                                                 &antenna1_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("antenna2", 0,
                                                 &antenna2_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("shape", 0,
                                                 &shape_ptr));
        OP_REQUIRES_OK(ctx, op_data.get_tensor("ant_jones", 0,
                                                 &ant_jones_ptr));
        bool have_complex_phase = op_data.get_tensor("complex_phase", 0,
                                                 &complex_phase_ptr).ok();
        OP_REQUIRES_OK(ctx, op_data.get_tensor("sgn_brightness", 0,
                                                 &sgn_brightness_ptr));
        bool have_base = op_data.get_tensor("base_coherencies", 0,
                                                 &base_coherencies_ptr).ok();

        // Dummy variables to handle the absence of inputs
        const tf::Tensor dummy_phase(tf::DataTypeToEnum<CT>::value, {1});
        const tf::Tensor dummy_base(tf::DataTypeToEnum<CT>::value, {1,1,1});

        auto time_index = time_index_ptr->tensor<int,1>();
        auto antenna1 = antenna1_ptr->tensor<int,1>();
        auto antenna2 = antenna2_ptr->tensor<int,1>();
        auto shape = shape_ptr->tensor<FT, 3>();
        auto ant_jones = ant_jones_ptr->tensor<CT, 5>();
        auto sgn_brightness = sgn_brightness_ptr->tensor<tf::int8, 2>();
        auto complex_phase = have_complex_phase ?
                        complex_phase_ptr->flat<CT>() :
                        dummy_phase.flat<CT>();
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
                    CT a0 = ant_jones(src, time, ant1, chan, 0);
                    CT a1 = ant_jones(src, time, ant1, chan, 1);
                    CT a2 = ant_jones(src, time, ant1, chan, 2);
                    CT a3 = ant_jones(src, time, ant1, chan, 3);

                    // Multiply shape value into antenna1 jones
                    const FT & s = shape(src, vrow, chan);

                    a0 = s*a0;
                    a1 = s*a1;
                    a2 = s*a2;
                    a3 = s*a3;

                    // Now multiply in the complex phase if we have it
                    if(have_complex_phase)
                    {
                        // complex_phase index is flat because it may be scalar
                        const int index = (src*nvrow + vrow)*nchan + chan;
                        const CT & cp = complex_phase(index);

                        a0 = cp*a0;
                        a1 = cp*a1;
                        a2 = cp*a2;
                        a3 = cp*a3;
                    }

                    // Conjugate transpose of antenna 2 jones with shape factor
                    const CT b0 = std::conj(ant_jones(src, time, ant2, chan, 0));
                    const CT b1 = std::conj(ant_jones(src, time, ant2, chan, 2));
                    const CT b2 = std::conj(ant_jones(src, time, ant2, chan, 1));
                    const CT b3 = std::conj(ant_jones(src, time, ant2, chan, 3));


                    FT sign = sgn_brightness(src, time);

                    // Multiply jones matrices and accumulate them
                    // in the sum terms
                    s0 += sign*(a0*b0 + a1*b2);
                    s1 += sign*(a0*b1 + a1*b3);
                    s2 += sign*(a2*b0 + a3*b2);
                    s3 += sign*(a2*b1 + a3*b3);
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
