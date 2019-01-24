#ifndef RIME_FEED_ANGLE_OP_CPU_H
#define RIME_FEED_ANGLE_OP_CPU_H

#include "feed_angle_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ANGLE_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the FeedAngle op for CPUs
template <typename FT, typename CT>
class FeedAngle<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string feed_type;

public:
    explicit FeedAngle(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("feed_type", &feed_type));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_feed_angle = context->input(0);
        const auto & in_parallactic_angle = context->input(1);

        int ntime = in_parallactic_angle.dim_size(0);
        int na = in_parallactic_angle.dim_size(1);

        // Allocate output tensors
        // Allocate space for output tensor 'feed_angle_rotation'
        tf::Tensor * feed_angle_rotation_ptr = nullptr;
        tf::TensorShape feed_angle_rotation_shape = tf::TensorShape({
            ntime, na, 4 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, feed_angle_rotation_shape, &feed_angle_rotation_ptr));

        // Extract Eigen tensors
        auto feed_angle = in_feed_angle.tensor<FT, 1>();
        auto parallactic_angle = in_parallactic_angle.tensor<FT, 2>();
        auto output = feed_angle_rotation_ptr->tensor<CT, 3>();

        if(feed_type == "linear")
        {
            #pragma omp parallel for collapse(2)
            for(int time=0; time < ntime; ++time)
            {
                for(int ant=0; ant < na; ++ant)
                {
                    int angle = feed_angle(ant) + parallactic_angle(time, ant);
                    FT f_sin = std::sin(angle);
                    FT f_cos = std::cos(angle);

                    output(time, ant, 0) = CT(f_cos, 0.0);
                    output(time, ant, 1) = CT(f_sin, 0.0);
                    output(time, ant, 2) = CT(-f_sin, 0.0);
                    output(time, ant, 3) = CT(f_cos, 0.0);
                }

            }
        }
        else if(feed_type == "circular")
        {
            #pragma omp parallel for collapse(2)
            for(int time=0; time < ntime; ++time)
            {
                for(int ant=0; ant < na; ++ant)
                {
                    int angle = feed_angle(ant) + parallactic_angle(time, ant);
                    FT f_sin = std::sin(angle);
                    FT f_cos = std::cos(angle);

                    output(time, ant, 0) = CT(f_cos, -f_sin);
                    output(time, ant, 1) = CT(0.0, 0.0);
                    output(time, ant, 2) = CT(0.0, 0.0);
                    output(time, ant, 3) = CT(f_cos, f_sin);
                }
            }
        }
        else
        {
            // Induce failure
            OP_REQUIRES_OK(context, tf::Status(tf::errors::InvalidArgument(
                "Invalid feed type '", feed_type, "'. "
                "Must be 'linear' or 'circular'")));
        }
    }
};

MONTBLANC_FEED_ANGLE_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ANGLE_OP_CPU_H
