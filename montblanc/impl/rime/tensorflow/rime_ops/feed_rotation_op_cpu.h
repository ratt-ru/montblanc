#ifndef RIME_FEED_ROTATION_OP_CPU_H
#define RIME_FEED_ROTATION_OP_CPU_H

#include "feed_rotation_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

MONTBLANC_NAMESPACE_BEGIN
MONTBLANC_FEED_ROTATION_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the FeedRotation op for CPUs
template <typename FT, typename CT>
class FeedRotation<CPUDevice, FT, CT> : public tensorflow::OpKernel
{
private:
    std::string feed_type;

public:
    explicit FeedRotation(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context)

    {
        OP_REQUIRES_OK(context, context->GetAttr("feed_type", &feed_type));
    }

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_parallactic_angle_sin = context->input(0);
        const auto & in_parallactic_angle_cos = context->input(1);


        int ntime = in_parallactic_angle_sin.dim_size(0);
        int na = in_parallactic_angle_sin.dim_size(1);
        int npa = ntime*na;


        // Allocate output tensors
        // Allocate space for output tensor 'feed_rotation'
        tf::Tensor * feed_rotation_ptr = nullptr;
        tf::TensorShape feed_rotation_shape = tf::TensorShape(
            { ntime, na, FEED_ROTATION_NPOL });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, feed_rotation_shape, &feed_rotation_ptr));

        // Extract Eigen tensors
        auto feed_rotation = feed_rotation_ptr->shaped<CT, 2>(
                                    {npa, FEED_ROTATION_NPOL});
        auto pa_sin = in_parallactic_angle_sin.flat<FT>();
        auto pa_cos = in_parallactic_angle_cos.flat<FT>();

        if(feed_type == "linear") {
            #pragma omp parallel
            for(int pa=0; pa < pa_sin.size(); ++pa)
            {
                feed_rotation(pa, 0) = CT(pa_cos(pa), 0);
                feed_rotation(pa, 1) = CT(pa_sin(pa), 0);
                feed_rotation(pa, 2) = CT(-pa_sin(pa), 0);
                feed_rotation(pa, 3) = CT(pa_cos(pa), 0);
            }
        } else if(feed_type == "circular") {
            #pragma omp parallel
            for(int pa=0; pa < pa_sin.size(); ++pa)
            {
                // exp(i*pa) == cos(pa) + i*sin(pa)
                // exp(-i*pa) == cos(pa) - i*sin(pa)
                feed_rotation(pa, 0) = { pa_cos(pa), -pa_sin(pa) }; // exp(-i*pa)
                feed_rotation(pa, 1) = { 0, 0 };
                feed_rotation(pa, 2) = { 0, 0 };
                feed_rotation(pa, 3) = { pa_cos(pa), pa_sin(pa) }; // exp(i*pa)
            }
        } else {
            // Induce failure
            OP_REQUIRES_OK(context, tf::Status(tf::errors::InvalidArgument(
                "Invalid feed type '", feed_type, "'. "
                "Must be 'linear' or 'circular'")));
        }
    }
};

MONTBLANC_FEED_ROTATION_NAMESPACE_STOP
MONTBLANC_NAMESPACE_STOP

#endif // #ifndef RIME_FEED_ROTATION_OP_CPU_H